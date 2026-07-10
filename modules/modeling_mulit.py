from __future__ import absolute_import, division, print_function

import logging
import math

import torch
import torch.nn.functional as F
from torch import nn

from modules.backbone_adapter import (
    build_eva_clip_backbone,
    get_eva_clip_backbone_spec,
    load_eva_clip_pretrained,
)
from modules.module_clip import CLIP, convert_weights
from modules.module_cross import CrossConfig
from modules.module_cross import Transformer as TransformerClip
from modules.spatial_enhancer import SpatialEnhancer
from modules.until_module import (
    MILNCELoss_BoF,
    MultiPositiveCrossEn,
    PreTrainedModel,
    allgather_no_grad,
    allgather_with_grad,
)
from query_models.module_sap import SemanticAnchorProbing

try:
    from prob_models.pie_model import PIENet
    from prob_models.tensor_utils import l2_normalize, sample_gaussian_tensors
    from prob_models.uncertainty_module import (
        UncertaintyAdaNorm,
        UncertaintyModuleImage,
        UncertaintyModuleText,
        UncertaintyModuleTextMamba,
    )
except Exception as e:
    raise EnvironmentError("Failed to import probabilistic modules. Please check dependencies.") from e

logger = logging.getLogger(__name__)


class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @staticmethod
    def _validate_checkpoint_backbone(state_dict, backbone_type):
        """Reject checkpoints that explicitly identify a different visual backbone.

        Checkpoints containing only UATVR layers do not carry enough information to
        identify a backbone and are intentionally allowed; the selected backbone's
        base weights will be supplied by the normal initialization path.
        """
        openai_markers = tuple(
            key
            for key in state_dict
            if key.startswith("clip.visual.conv1.") or key.startswith("clip.visual.transformer.")
        )
        eva_markers = tuple(
            key
            for key in state_dict
            if key.startswith("clip.visual.patch_embed.")
            or key.startswith("clip.visual.blocks.")
            or key.startswith("clip.visual.pos_embed")
        )

        if openai_markers and eva_markers:
            raise ValueError(
                "Checkpoint contains mixed backbone keys for openai_clip and eva_clip: "
                f"openai_clip={openai_markers[:3]}, eva_clip={eva_markers[:3]}"
            )

        detected_type = "openai_clip" if openai_markers else "eva_clip" if eva_markers else None
        if detected_type is not None and detected_type != backbone_type:
            marker_keys = openai_markers if detected_type == "openai_clip" else eva_markers
            raise ValueError(
                f"backbone_type={backbone_type} is incompatible with detected checkpoint backbone "
                f"{detected_type}; marker_keys={marker_keys[:5]}"
            )

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):
        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None:
            state_dict = {}
        backbone_type = getattr(task_config, "backbone_type", "openai_clip")
        cls._validate_checkpoint_backbone(state_dict, backbone_type)
        clip_state_dict = None
        if backbone_type == "openai_clip":
            pretrained_clip_name = getattr(task_config, "pretrained_clip_name", "ViT-B/32")
            clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
            for key, val in clip_state_dict.items():
                new_key = "clip." + key
                if new_key not in state_dict:
                    state_dict[new_key] = val.clone()
        elif backbone_type == "eva_clip":
            pass
        else:
            raise ValueError(f"Unsupported backbone_type={backbone_type}")

        # 只加载 config，不需加载 weights（cross 模块随机初始化训练）。
        # 传 state_dict={} 跳过 get_config 中"权重文件不存在"的检查。
        cross_config, _ = CrossConfig.get_config(
            cross_model_name, cache_dir, type_vocab_size, state_dict={}, task_config=task_config
        )

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)  # -----------

        ## ===> Initialization trick [HARD CODE]
        if backbone_type == "eva_clip":
            has_eva_clip = any(
                key.startswith("clip.visual.patch_embed.")
                or key.startswith("clip.visual.blocks.")
                or key.startswith("clip.visual.pos_embed")
                for key in state_dict
            )
            if not has_eva_clip:
                load_eva_clip_pretrained(
                    model.clip,
                    backbone_name=getattr(task_config, "backbone_name", "EVA02-CLIP-B-16"),
                    backbone_path=getattr(task_config, "backbone_path", None),
                    eva_clip_root=getattr(task_config, "eva_clip_root", None),
                    use_xattn=getattr(task_config, "eva_clip_use_xattn", False),
                )
            model.prepare_seqtransf_init_from_backbone(state_dict)
        elif model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros is not None:
                    cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros is not None:
                    cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        if backbone_type == "openai_clip" and model.sim_header == "seqTransf":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        # CLIP positional_embedding is [77, dim]; our frame_position_embeddings
                        # may be larger (e.g. [128, dim]) after the max_position_embeddings fix.
                        # Copy CLIP weights into the first 77 rows; leave the rest randomly
                        # initialised (from the model's own nn.Embedding init).
                        clip_pos_len = val.shape[0]
                        model_pos_len = model.frame_position_embeddings.weight.shape[0]
                        if model_pos_len > clip_pos_len:
                            new_val = model.frame_position_embeddings.weight.data.clone()
                            new_val[:clip_pos_len] = val.clone()
                            state_dict["frame_position_embeddings.weight"] = new_val
                        else:
                            state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model


def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)


def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(
                source_config,
                "Set {}.{}: {}.".format(target_name, target_attr_name, getattr(target_config, target_attr_name)),
            )
    return target_config


def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


class UATVR(CLIP4ClipPreTrainedModel):
    HYGIENE_FROZEN_PARAMETER_PREFIXES = (
        "sap.",
        "pie_net_text.",
        "uncertain_net_text.",
        "ada_norm_text.",
        "spatial_enhancer.",
    )

    @staticmethod
    def _validate_backbone_contract(backbone, embed_dim, d_model):
        output_dim = getattr(backbone, "output_dim", None)
        dimensions = f"output_dim={output_dim}, embed_dim={embed_dim}, d_model={d_model}"
        if output_dim != embed_dim or output_dim != d_model:
            raise ValueError(f"Backbone dimension contract mismatch: {dimensions}")

        for capability in ("supports_text_hidden", "supports_visual_hidden"):
            if not getattr(backbone, capability, False):
                raise ValueError(
                    f"Backbone capability contract mismatch: {capability}=False; "
                    "the current UATVR path requires projected token/patch hidden states."
                )

    def __init__(self, cross_config, clip_state_dict, task_config):
        super(UATVR, self).__init__(cross_config)
        self.task_config = task_config

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self.loose_type = False
        if check_attr("loose_type", self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        self.backbone_type = getattr(task_config, "backbone_type", "openai_clip")
        if self.backbone_type == "openai_clip":
            # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
            assert "visual.proj" in clip_state_dict, "Only ViT-based CLIP is supported"
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")]
            )
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size

            embed_dim = clip_state_dict["text_projection"].shape[1]
            context_length = clip_state_dict["positional_embedding"].shape[0]
            vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
            transformer_width = clip_state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64
            transformer_layers = len(
                set(k.split(".")[2] for k in clip_state_dict if k.startswith("transformer.resblocks"))
            )
        elif self.backbone_type == "eva_clip":
            spec = get_eva_clip_backbone_spec(
                getattr(task_config, "backbone_name", "EVA02-CLIP-B-16"),
                getattr(task_config, "eva_clip_root", None),
            )
            vision_width = spec.vision_width
            vision_layers = spec.vision_layers
            vision_patch_size = spec.vision_patch_size
            image_resolution = spec.image_resolution
            embed_dim = spec.embed_dim
            context_length = spec.context_length
            vocab_size = spec.vocab_size
            transformer_width = spec.transformer_width
            transformer_heads = spec.transformer_heads
            transformer_layers = spec.transformer_layers
        else:
            raise ValueError(f"Unsupported backbone_type={self.backbone_type}")

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = "2d"
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        if self.backbone_type == "openai_clip":
            # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
            self.clip = CLIP(
                embed_dim,
                image_resolution,
                vision_layers - cut_top_layer,
                vision_width,
                vision_patch_size,
                context_length,
                vocab_size,
                transformer_width,
                transformer_heads,
                transformer_layers - cut_top_layer,
                linear_patch=self.linear_patch,
            ).float()

            for key in ["input_resolution", "context_length", "vocab_size"]:
                if key in clip_state_dict:
                    del clip_state_dict[key]

            convert_weights(self.clip)
            # <=== End of CLIP Encoders
        else:
            self.clip = build_eva_clip_backbone(
                backbone_name=getattr(task_config, "backbone_name", "EVA02-CLIP-B-16"),
                backbone_path=getattr(task_config, "backbone_path", None),
                eva_clip_root=getattr(task_config, "eva_clip_root", None),
                use_xattn=getattr(task_config, "eva_clip_use_xattn", False),
                load_pretrained=False,
            ).float()
            self._validate_backbone_contract(self.clip, embed_dim=embed_dim, d_model=transformer_width)

        self.sim_header = "meanP"
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        # 确保位置嵌入表覆盖最长序列：
        #   text branch  : max_words + extra_text_cls_num  (e.g. 32+2 = 34)
        #   visual branch: max_frames + extra_cls_frame_num  (e.g. 12+2 = 14)
        # cross-base config 默认 128 已覆盖；仅当加载的 config 更小时才向上覆盖。
        _extra_cls = getattr(task_config, "extra_text_cls_num", 2)
        _min_pos = max(
            task_config.max_words + _extra_cls,
            task_config.max_frames + getattr(task_config, "extra_video_cls_num", 2),
        )
        cross_config.max_position_embeddings = max(cross_config.max_position_embeddings, _min_pos)

        if self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(
                cross_config.max_position_embeddings, cross_config.hidden_size
            )
            self.word_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(
                width=transformer_width,
                layers=self.task_config.cross_num_hidden_layers,
                heads=transformer_heads,
            )

        # 视频侧 PIENet 已移除：SAP 的 Dirichlet 模态概率聚合替代了 PIENet 的 cross-modal attention

        self.pie_net_text = PIENet(1, embed_dim, embed_dim, embed_dim // 2)
        # Text-side uncertainty head can be configured (default keeps previous lightweight head).
        uncert_text_head = getattr(self.task_config, "uncertainty_text_head", "image")
        if uncert_text_head == "text":
            self.uncertain_net_text = UncertaintyModuleText(embed_dim, embed_dim, embed_dim // 2)
        elif uncert_text_head == "mamba":
            self.uncertain_net_text = UncertaintyModuleTextMamba(embed_dim, embed_dim, embed_dim // 2)
        else:
            self.uncertain_net_text = UncertaintyModuleImage(embed_dim, embed_dim, embed_dim // 2)

        self.use_ada_norm = getattr(self.task_config, "use_ada_norm", False)
        if self.use_ada_norm:
            self.ada_norm_text = UncertaintyAdaNorm(embed_dim)

        self.n_video_samples = self.task_config.n_video_embeddings  # numbers sampling from video distribution 7
        self.n_text_samples = self.task_config.n_text_embeddings  # numbers sampling from text distribution 7

        rope_mode = getattr(self.task_config, 'rope_mode', 'none')
        if rope_mode != 'none':
            self.spatial_enhancer = SpatialEnhancer(
                embed_dim=transformer_width, num_heads=transformer_heads, rope_mode=rope_mode
            )
        else:
            self.spatial_enhancer = None

        # Token-importance heads for WTI matching
        self.text_weight_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True), nn.Linear(transformer_width, 1)
        )
        self.video_weight_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True), nn.Linear(transformer_width, 1)
        )
        self.qc_sap_text_proj = nn.Linear(transformer_width, transformer_width)
        self.qc_sap_anchor_proj = nn.Linear(transformer_width, transformer_width)

        # Semantic Anchor Probing (SAP): decomposes video into semantic anchors
        # with per-anchor uncertainty for the probabilistic pipeline.
        self.num_anchors = getattr(self.task_config, "num_queries", 16)
        self.sap = SemanticAnchorProbing(
            d_model=transformer_width, num_anchors=self.num_anchors,
            nhead=transformer_heads, num_layers=2,
            log_sigma_min=getattr(task_config, "log_sigma_min", None),
            log_sigma_max=getattr(task_config, "log_sigma_max", None),
        )

        # Video-ColBERT style expansion tokens: 可学习 token 拼接在 anchor 序列旁，
        # 给 PIENet/WTI 更多聚合自由度。
        self.num_expansion_tokens = getattr(self.task_config, "num_expansion_tokens", 0)
        if self.num_expansion_tokens > 0:
            self.expansion_tokens = nn.Parameter(
                torch.zeros(1, self.num_expansion_tokens, transformer_width)
            )
            nn.init.normal_(self.expansion_tokens, std=0.02)

        # Loss functions
        self.loss_fct = MultiPositiveCrossEn()
        self.loss_MIL_fct = MILNCELoss_BoF()

        # Loss weights
        self.w_mil = getattr(self.task_config, "w_mil", 1e-2)
        self.w_evidential = getattr(self.task_config, "w_evidential", 1e-2)
        self.w_neg_reg = getattr(self.task_config, "w_neg_reg", 1e-2)
        self.w_uncertainty_reg = getattr(self.task_config, "w_uncertainty_reg", 1e-3)
        self.w_orth = getattr(self.task_config, "w_orth", 0.0)
        self.final_score_mode = getattr(self.task_config, "final_score_mode", "wti")
        self.lambda_prob = getattr(self.task_config, "lambda_prob", 0.0)
        self.lambda_anchor = getattr(self.task_config, "lambda_anchor", 0.0)
        self.lambda_qc_sap = getattr(self.task_config, "lambda_qc_sap", 0.0)
        self.qc_sap_temperature = getattr(self.task_config, "qc_sap_temperature", 0.1)
        self.use_explicit_hard_negative_loss = getattr(self.task_config, "use_explicit_hard_negative_loss", False)
        self.w_hard_negative = getattr(self.task_config, "w_hard_negative", 5e-2)
        self.use_uacl_intra_alignment = getattr(self.task_config, "use_uacl_intra_alignment", False)
        self.w_uacl_intra = getattr(self.task_config, "w_uacl_intra", 1e-2)
        self.w_uacl_kl = getattr(self.task_config, "w_uacl_kl", 1e-4)
        self.uacl_temperature = getattr(self.task_config, "uacl_temperature", 0.07)
        self.uacl_sample_strategy = getattr(self.task_config, "uacl_sample_strategy", "closest")
        self.experiment_profile = getattr(self.task_config, "experiment_profile", "default")

        # 退火系数：默认关闭；warmup_epochs > 0 时前 N 个 epoch 线性增大到 1.0
        self.anneal_warmup_epochs = getattr(self.task_config, "anneal_warmup_epochs", 0)
        self._current_epoch = 0  # 由外部 train_loop 更新

        # 不确定性训练模式：evidential / nig_mil(deprecated) / none
        self.uncertainty_mode = getattr(self.task_config, "uncertainty_mode", "none")
        self.loss_activations = self.resolve_loss_activations(self.task_config)

        # Optional clamp for log-variance to stabilize sampling/KL terms
        self.log_sigma_min = getattr(self.task_config, "log_sigma_min", None)
        self.log_sigma_max = getattr(self.task_config, "log_sigma_max", None)

        # extra class token num
        self.extra_cls_frame_num = self.task_config.extra_video_cls_num
        self.extra_cls_text_num = self.task_config.extra_text_cls_num

        show_log(task_config, "CLIP UATVR Model ......")
        show_log(task_config, "\t Extra video Class token number: {}".format(self.extra_cls_frame_num))
        show_log(task_config, "\t Extra text Class token number: {}".format(self.extra_cls_text_num))
        show_log(task_config, "\t Number of video sampling probabilistic embeddings: {}".format(self.n_video_samples))
        show_log(task_config, "\t Number of text sampling probabilistic embeddings: {}".format(self.n_text_samples))
        show_log(
            task_config,
            "\t Final score mode: {} (lambda_prob={}, lambda_anchor={}, lambda_qc_sap={}, qc_temp={})".format(
                self.final_score_mode,
                self.lambda_prob,
                self.lambda_anchor,
                self.lambda_qc_sap,
                self.qc_sap_temperature,
            ),
        )

        # 诊断统计步数计数器（每 N 步采集一次，减少显存和计算开销）
        self._diag_step = 0
        self._diag_interval = getattr(task_config, "diag_interval", 10)
        self._diag_chain = {}
        self._prob_chain = {}
        self._aux_chain = {}

        self.apply(self.init_weights)
        frozen_count = self.freeze_inactive_parameters_for_profile()
        if frozen_count > 0:
            show_log(
                task_config,
                "\t Frozen hygiene-only auxiliary parameters: {}".format(frozen_count),
            )

    def prepare_seqtransf_init_from_backbone(self, state_dict):
        if self.sim_header != "seqTransf":
            return

        contain_frame_position = any(key.find("frame_position_embeddings") > -1 for key in state_dict.keys())
        if not contain_frame_position and hasattr(self, "frame_position_embeddings"):
            source_pos = getattr(self.clip, "positional_embedding", None)
            if source_pos is not None:
                source_pos = source_pos.detach()
                clip_pos_len = source_pos.shape[0]
                model_pos_len = self.frame_position_embeddings.weight.shape[0]
                if model_pos_len > clip_pos_len:
                    new_val = self.frame_position_embeddings.weight.data.clone()
                    new_val[:clip_pos_len] = source_pos.clone()
                    state_dict["frame_position_embeddings.weight"] = new_val
                else:
                    state_dict["frame_position_embeddings.weight"] = source_pos[:model_pos_len].clone()

        if not hasattr(self, "transformerClip") or not hasattr(self.clip, "transformer"):
            return

        for key, val in self.clip.transformer.state_dict().items():
            if not key.startswith("resblocks."):
                continue
            num_layer = int(key.split(".")[1])
            if num_layer >= self.task_config.cross_num_hidden_layers:
                continue
            target_key = "transformerClip." + key
            if target_key not in state_dict:
                state_dict[target_key] = val.detach().clone()

    @staticmethod
    def _flatten_video_input(video):
        video = torch.as_tensor(video).float()
        assert video.dim() == 7, f"Expected 7D video tensor, got shape={tuple(video.shape)}"
        b, pair, num_frames, clips_per_frame, channel, h, w = video.shape
        video = video.view(b * pair * num_frames * clips_per_frame, channel, h, w)
        return video, num_frames * clips_per_frame

    @staticmethod
    def resolve_video_group_ids(
        video_group_id, local_batch, device, task_config
    ):
        if video_group_id is not None:
            video_group_id = torch.as_tensor(video_group_id)
            integer_dtypes = {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
            if video_group_id.dtype not in integer_dtypes:
                raise ValueError(
                    "video_group_id must use an integer dtype, "
                    f"got dtype={video_group_id.dtype}"
                )
            resolved = video_group_id.reshape(-1).to(
                device=device, dtype=torch.long
            )
            if resolved.numel() != int(local_batch):
                raise ValueError(
                    f"video_group_id length={resolved.numel()} does not match "
                    f"local batch={int(local_batch)}"
                )
            return resolved
        if getattr(task_config, "datatype", "msrvtt") == "msrvtt":
            raise ValueError("trusted-v1 MSRVTT training requires video_group_id")
        rank = getattr(task_config, "rank", None)
        if rank is None:
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                rank = torch.distributed.get_rank()
            else:
                rank = 0
        return (
            torch.arange(local_batch, device=device, dtype=torch.long)
            + int(rank) * 10_000_000
        )

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        video,
        video_mask=None,
        video_group_id=None,
        sample_index=None,
        hard_video=None,
        hard_video_mask=None,
        hard_valid=None,
    ):
        if self.training:
            video_group_id = self.resolve_video_group_ids(
                video_group_id,
                local_batch=input_ids.size(0),
                device=input_ids.device,
                task_config=self.task_config,
            )
        # (B 1 32)  (B 1 132) (B 1 32)
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        # video_mask: [B, pair, num_frames] -> [B*pair, num_frames]
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video, video_len = self._flatten_video_input(video)

        sequence_output, text_token = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)

        visual_cls, visual_hidden = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_len)

        if self.training:
            hard_visual_cls = None
            if self.loss_activations["hard_negative"] and hard_video is not None and hard_video_mask is not None:
                hard_video_mask = hard_video_mask.view(-1, hard_video_mask.shape[-1])
                hard_video, hard_video_len = self._flatten_video_input(hard_video)
                hard_visual_cls, _hard_visual_hidden = self.get_visual_output(
                    hard_video,
                    hard_video_mask,
                    shaped=True,
                    video_frame=hard_video_len,
                )

            loss = 0.0
            res = self.get_similarity_logits(
                sequence_output,
                text_token,
                visual_cls,
                visual_hidden,
                attention_mask,
                video_mask,
                shaped=True,
                loose_type=self.loose_type,
                video_group_id=video_group_id,
                hard_visual_output=hard_visual_cls,
                hard_video_mask=hard_video_mask,
                hard_valid=hard_valid,
            )
            sim_matrix = res["retrieve_logits"]
            global_group_ids = res["video_group_id"]
            sim_loss_t2v = self.loss_fct(
                sim_matrix, global_group_ids, global_group_ids
            )
            sim_loss_v2t = self.loss_fct(
                sim_matrix.T, global_group_ids, global_group_ids
            )
            sim_loss = (sim_loss_t2v + sim_loss_v2t) / 2
            positive_mask = global_group_ids[:, None].eq(
                global_group_ids[None, :]
            )
            positive_counts = positive_mask.sum(dim=1).float()
            unique_count = torch.unique(global_group_ids).numel()

            loss += sim_loss
            loss += res["MIL_loss"]
            loss += res["evidential_loss"]
            loss += res["neg_reg_loss"]
            loss += res["orth_loss"]
            loss += res["hard_negative_loss"]
            loss += res["uacl_intra_loss"]
            loss += res["uacl_kl_loss"]

            loss_dict = {
                "total": loss,
                "sim_loss": sim_loss,
                "mcsoft_loss": res["MIL_loss"],
                "evidential_loss": res["evidential_loss"],
                "neg_reg_loss": res["neg_reg_loss"],
                "orth_loss": res["orth_loss"],
                "hard_negative_loss": res["hard_negative_loss"],
                "uacl_intra_loss": res["uacl_intra_loss"],
                "uacl_kl_loss": res["uacl_kl_loss"],
                "unique_video_count": sim_matrix.new_tensor(float(unique_count)),
                "duplicate_sample_count": sim_matrix.new_tensor(
                    float(global_group_ids.numel() - unique_count)
                ),
                "mean_positive_count": positive_counts.mean(),
            }
            return loss_dict
        else:  # for inference
            return None

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])  # B*num 32
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])  # B*num 32
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])  # B*num 32

        bs_pair = input_ids.size(0)  # B 2 32
        sequence_output, text_token = self.clip.encode_text(input_ids, return_hidden=True)
        sequence_output = sequence_output.float()
        text_token = text_token.float()

        sequence_output = sequence_output.view(bs_pair, -1, sequence_output.size(-1))
        text_token = text_token.view(bs_pair, -1, text_token.size(-1))

        return sequence_output, text_token

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            assert video.dim() == 7, f"Expected 7D video tensor, got shape={tuple(video.shape)}"
            b, pair, num_frames, clips_per_frame, channel, h, w = video.shape
            video = video.view(b * pair * num_frames * clips_per_frame, channel, h, w)
            video_frame = num_frames * clips_per_frame

        bs_pair = video_mask.size(0)  # video: bs*2*frame       video_frame:2*frame   video_mask:bs 2 frame
        # 使用 return_hidden=True 获取双输出: CLS + all tokens
        visual_cls, visual_hidden = self.clip.encode_image(video, return_hidden=True, video_frame=video_frame)
        visual_cls = visual_cls.float()
        visual_hidden = visual_hidden.float()

        # visual_cls: [bs*pair*bs*ts, 512] -> [bs_pair, video_frame, 512]
        visual_cls = visual_cls.view(bs_pair, -1, visual_cls.size(-1))

        # visual_hidden: [bs*pair*bs*ts, 197, 512] -> [bs_pair, video_frame, 197, 512]
        visual_hidden = visual_hidden.view(bs_pair, video_frame, visual_hidden.size(-2), visual_hidden.size(-1))

        if hasattr(self, "spatial_enhancer") and self.spatial_enhancer is not None:
            B, T, L, D = visual_hidden.shape
            if L > 1:
                spatial_tokens = visual_hidden[:, :, 1:, :]
                spatial_len = spatial_tokens.size(2)
                side = int(math.sqrt(spatial_len))
                if side * side == spatial_len:
                    if self.spatial_enhancer.rope_mode == '3d':
                        # [B, T, HW, D] -> [B, D, T, H, W]
                        spatial_tokens = spatial_tokens.reshape(B, T, side, side, D).permute(0, 4, 1, 2, 3)
                        spatial_tokens = self.spatial_enhancer(spatial_tokens)
                        spatial_tokens = spatial_tokens.permute(0, 2, 3, 4, 1).reshape(B, T, spatial_len, D)
                    else:
                        # [B, T, HW, D] -> [B*T, D, H, W]
                        spatial_tokens = spatial_tokens.reshape(B * T, side, side, D).permute(0, 3, 1, 2)
                        spatial_tokens = self.spatial_enhancer(spatial_tokens)
                        spatial_tokens = spatial_tokens.permute(0, 2, 3, 1).reshape(B, T, spatial_len, D)
                    visual_hidden = torch.cat([visual_hidden[:, :, :1, :], spatial_tokens], dim=2)

        return visual_cls, visual_hidden

    def get_sequence_visual_output(
        self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1
    ):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            # video_mask = video_mask.view(-1, video_mask.shape[-1])        # bs 2 num_frame  =>  bs*2 num_frame

            video = torch.as_tensor(video).float()
            assert video.dim() == 7, f"Expected 7D video tensor, got shape={tuple(video.shape)}"
            b, pair, num_frames, clips_per_frame, channel, h, w = video.shape
            video = video.view(b * pair * num_frames * clips_per_frame, channel, h, w)
            video_frame = num_frames * clips_per_frame

        sequence_output, hidden_word = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=False)
        visual_output, visual_output_all = self.get_visual_output(
            video, video_mask, shaped=True, video_frame=video_frame
        )
        return sequence_output, hidden_word, visual_output, visual_output_all

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.0
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(
        self,
        visual_output,
        video_mask,
    ):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.0] = 1.0
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _loose_similarity(
        self,
        sequence_output,
        text_token,
        visual_output,
        visual_output_hidden,
        attention_mask,
        video_mask,
        sim_header="seqTransf",
        video_group_id=None,
        hard_visual_output=None,
        hard_video_mask=None,
        hard_valid=None,
    ):
        sequence_output, visual_output = (
            sequence_output.contiguous(),
            visual_output.contiguous(),
        )  # visual_output [B, T, D]
        visual_output_hidden = visual_output_hidden.contiguous()  # [B, T, 197, D]
        frame_num = visual_output.size(1)
        word_num = text_token.size(1)

        if sim_header == "seqTransf":
            visual_output_original = visual_output

            extra_token_num = self.extra_cls_frame_num
            seq_length = visual_output.size(1) + extra_token_num
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            frame_position_embeddings[:, 0 : visual_output.size(1), :] += visual_output
            visual_output = frame_position_embeddings

            tempo_mask = torch.cat(
                [video_mask, torch.ones(visual_output.size(0), extra_token_num).to(visual_output.device)], axis=1
            )
            extended_video_mask = (1.0 - tempo_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, tempo_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2).contiguous()
            visual_output[:, : visual_output_original.size(1), :] += visual_output_original
            video_mask = tempo_mask

            text_original = text_token
            extra_text_num = self.extra_cls_text_num
            seq_text_length = extra_text_num + text_token.size(1)
            position_ids_text = torch.arange(seq_text_length, dtype=torch.long, device=text_token.device)
            position_ids_text = position_ids_text.unsqueeze(0).expand(text_token.size(0), -1)
            word_position_embeddings = self.word_position_embeddings(position_ids_text)
            word_position_embeddings[:, 0 : text_token.size(1), :] += text_token
            text_token = word_position_embeddings

            tempo_mask_ = torch.cat(
                [attention_mask, torch.ones(text_token.size(0), extra_text_num).to(text_token.device)], axis=1
            )
            extended_text_mask = (1.0 - tempo_mask_.unsqueeze(1)) * -1000000.0
            extended_text_mask = extended_text_mask.expand(-1, tempo_mask_.size(1), -1)
            text_token = text_token.permute(1, 0, 2)
            text_token = self.transformerClip(text_token, extended_text_mask)
            text_token = text_token.permute(1, 0, 2).contiguous()
            text_token[:, : text_original.size(1), :] += text_original
            attention_mask = tempo_mask_

        # ===== SAP: rank-local，先于 allgather，避免 4D hidden 张量广播 =====
        B_vis = visual_output_hidden.size(0)
        T_vis = visual_output_hidden.size(1)
        S_vis = visual_output_hidden.size(2)
        spatial_tokens = visual_output_hidden.reshape(B_vis, T_vis * S_vis, -1)
        spatial_mask = video_mask[:, 0:frame_num].unsqueeze(-1).expand(-1, -1, S_vis).reshape(B_vis, -1)
        sap_out = self.sap(
            spatial_tokens.contiguous(),
            padding_mask=(spatial_mask == 0),
        )
        anchors = sap_out["anchors"]           # [B, K, D] decoder 输出
        mu_video = sap_out["mu_raw"]           # [B, D] 模态概率聚合均值 (L2 norm)
        logsigma_video = sap_out["logsigma"]   # [B, D] mixture log 方差
        epistemic_video = sap_out["epistemic_cont"]  # [B, K, D] 连续认知不确定性
        u_mode = sap_out["u_mode"]             # [B] 离散模态不确定性
        alpha_dir = sap_out["alpha_dir"]       # [B, K] Dirichlet 证据量

        if self.log_sigma_min is not None and self.log_sigma_max is not None:
            logsigma_video = torch.clamp(logsigma_video, min=float(self.log_sigma_min), max=float(self.log_sigma_max))

        # ===== DDP allgather：只广播紧凑张量，不再广播 visual_output_hidden =====
        positive_mask = None
        if self.training:
            video_group_id = allgather_no_grad(
                video_group_id.contiguous().reshape(-1), self.task_config
            )
            mu_video = allgather_with_grad(mu_video.contiguous(), self.task_config)
            logsigma_video = allgather_with_grad(
                logsigma_video.contiguous(), self.task_config
            )
            epistemic_video = allgather_with_grad(
                epistemic_video.contiguous(), self.task_config
            )
            u_mode = allgather_with_grad(u_mode.contiguous(), self.task_config)
            alpha_dir = allgather_with_grad(alpha_dir.contiguous(), self.task_config)
            if self.final_score_mode in {"wti_anchor_wti", "wti_qc_sap"}:
                anchors = allgather_with_grad(anchors.contiguous(), self.task_config)
            visual_output = allgather_with_grad(
                visual_output.contiguous(), self.task_config
            )
            video_mask = allgather_no_grad(
                video_mask.contiguous(), self.task_config
            )
            if hard_visual_output is not None and hard_video_mask is not None:
                hard_visual_output = allgather_with_grad(
                    hard_visual_output.contiguous(), self.task_config
                )
                hard_video_mask = allgather_no_grad(
                    hard_video_mask.contiguous(), self.task_config
                )
                if hard_valid is not None:
                    hard_valid = allgather_no_grad(
                        hard_valid.contiguous().reshape(-1), self.task_config
                    )
            sequence_output = allgather_with_grad(
                sequence_output.contiguous(), self.task_config
            )
            text_token = allgather_with_grad(
                text_token.contiguous(), self.task_config
            )
            attention_mask = allgather_no_grad(
                attention_mask.contiguous(), self.task_config
            )
            positive_mask = video_group_id[:, None].eq(video_group_id[None, :])

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        if hard_visual_output is not None:
            hard_visual_output = hard_visual_output / hard_visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        text_token = text_token / text_token.norm(dim=-1, keepdim=True)
        text_pooled = self._mean_pooling_for_similarity_sequence(
            text_token[:, 0:word_num, :].contiguous(), attention_mask[:, 0:word_num].contiguous()
        )
        text_pooled = text_pooled / text_pooled.norm(dim=-1, keepdim=True)

        # ===== Evidential 相似度：cosine × epistemic 置信折扣 =====
        ev_sim = self._evidential_similarity(mu_video, text_pooled, epistemic_video)

        active = self.loss_activations if self.training else None
        needs_text_samples = bool(self.training and (active["mil"] or active["uacl_intra"]))

        # ===== Text 侧概率建模（保持 Gaussian，非对称设计） =====
        prob_text = self.probabilistic_text(
            text_pooled,
            text_token[:, 0:word_num, :].contiguous(),
            attention_mask[:, 0:word_num].contiguous(),
            sample_embeddings=needs_text_samples,
        )
        prob_text_embedding = prob_text.get("embedding")
        prob_text_logsigma = prob_text["logsigma"]

        # ===== WTI：帧级 token 交互打分 =====
        logit_scale = self.clip.logit_scale.exp()
        wti_logits = self.weighted_token_wise_intersection(
            text_token, visual_output, attention_mask, video_mask
        ) * logit_scale

        prob_mu_logits = None
        anchor_wti_logits = None
        qc_sap_logits = None
        qc_sap_stats = {}
        if self.final_score_mode == "wti_prob_mu":
            prob_mu_logits = torch.matmul(prob_text["mean"], mu_video.t()) * logit_scale
        elif self.final_score_mode == "wti_anchor_wti":
            anchor_video_mask = torch.ones(
                anchors.size(0),
                anchors.size(1),
                dtype=video_mask.dtype,
                device=anchors.device,
            )
            anchor_wti_logits = self.weighted_token_wise_intersection(
                text_token,
                F.normalize(anchors, dim=-1),
                attention_mask,
                anchor_video_mask,
            ) * logit_scale
        elif self.final_score_mode == "wti_qc_sap":
            qc_sap_logits, qc_sap_stats = self.compute_query_conditioned_sap_logits(
                self.qc_sap_text_proj(text_pooled),
                self.qc_sap_anchor_proj(anchors),
                logit_scale=logit_scale,
                temperature=self.qc_sap_temperature,
                positive_mask=positive_mask,
            )
        weighted_logits, score_source = self.compose_final_retrieval_logits(
            wti_logits,
            final_score_mode=self.final_score_mode,
            lambda_prob=self.lambda_prob,
            lambda_anchor=self.lambda_anchor,
            lambda_qc_sap=self.lambda_qc_sap,
            prob_mu_logits=prob_mu_logits,
            anchor_wti_logits=anchor_wti_logits,
            qc_sap_logits=qc_sap_logits,
        )

        if self.training:
            bs = mu_video.size(0)
            n_video = self.n_video_samples
            n_text = self.n_text_samples
            dim = mu_video.size(-1)

            prob_video_embedding = None
            if active["mil"] or active["uacl_intra"]:
                prob_video_embedding = sample_gaussian_tensors(mu_video, logsigma_video, n_video)

            if active["mil"]:
                # MIL loss：从 N(gamma, logsigma) 采样做多实例对比
                prob_sim_v = torch.einsum(
                    "ad,bd->ab",
                    prob_video_embedding.view(-1, dim),
                    prob_text_embedding.view(-1, dim),
                )
                prob_sim_t = torch.einsum(
                    "ad,bd->ab",
                    prob_text_embedding.view(-1, dim),
                    prob_video_embedding.view(-1, dim),
                )
                MIL_loss = (self.loss_MIL_fct(prob_sim_v, bs, n_video, n_text)
                            + self.loss_MIL_fct(prob_sim_t, bs, n_video, n_text)) / 2
            else:
                MIL_loss = torch.tensor(0.0, device=mu_video.device)

            # Evidential NLL + neg_reg：仅在 uncertainty_mode=evidential 时启用
            if active["evidential"]:
                evidential_loss = self._evidential_nll_loss(ev_sim, alpha_dir)
            else:
                evidential_loss = torch.tensor(0.0, device=mu_video.device)

            if active["neg_reg"]:
                neg_reg_loss = self._evidential_neg_reg_loss(ev_sim)
            else:
                neg_reg_loss = torch.tensor(0.0, device=mu_video.device)

            # 退火系数：warmup_epochs <= 0 时关闭退火，直接使用完整权重
            if self.anneal_warmup_epochs <= 0:
                anneal_factor = 1.0
            else:
                anneal_factor = min(1.0, self._current_epoch / self.anneal_warmup_epochs)

            hard_negative_loss = torch.tensor(0.0, device=mu_video.device)
            hard_diag_mean = None
            hard_pos_gap = None
            if (
                active["hard_negative"]
                and hard_visual_output is not None
                and hard_video_mask is not None
            ):
                hard_wti_logits = self.weighted_token_wise_intersection(
                    text_token,
                    hard_visual_output,
                    attention_mask,
                    hard_video_mask,
                ) * logit_scale
                pos_diag = weighted_logits.diagonal()
                hard_diag = hard_wti_logits.diagonal()
                if hard_valid is None:
                    hard_valid = torch.ones_like(pos_diag, dtype=torch.bool)
                hard_negative_loss = self._hard_negative_infonce_loss(weighted_logits, hard_wti_logits, hard_valid)
                valid_hard = hard_valid.to(device=pos_diag.device, dtype=torch.bool)
                if bool(valid_hard.any().item()):
                    hard_diag_mean = float(hard_diag[valid_hard].detach().mean().item())
                    hard_pos_gap = float((pos_diag[valid_hard] - hard_diag[valid_hard]).detach().mean().item())

            uacl_intra_loss = torch.tensor(0.0, device=mu_video.device)
            uacl_kl_loss = torch.tensor(0.0, device=mu_video.device)
            uacl_text_loss = torch.tensor(0.0, device=mu_video.device)
            uacl_video_loss = torch.tensor(0.0, device=mu_video.device)
            if active["uacl_intra"]:
                if prob_video_embedding is None:
                    prob_video_embedding = sample_gaussian_tensors(mu_video, logsigma_video, n_video)
                text_aug = self._select_uacl_gaussian_sample(
                    prob_text["mean"],
                    prob_text_embedding,
                    strategy=self.uacl_sample_strategy,
                )
                video_aug = self._select_uacl_gaussian_sample(
                    mu_video,
                    prob_video_embedding,
                    strategy=self.uacl_sample_strategy,
                )
                uacl_text_loss = self._uacl_intra_contrastive_loss(
                    prob_text["mean"],
                    text_aug,
                    temperature=self.uacl_temperature,
                )
                uacl_video_loss = self._uacl_intra_contrastive_loss(
                    mu_video,
                    video_aug,
                    temperature=self.uacl_temperature,
                )
                uacl_intra_loss = (uacl_text_loss + uacl_video_loss) / 2.0
            if active["uacl_kl"]:
                uacl_kl_loss = (self._logvar_kl(prob_text_logsigma) + self._logvar_kl(logsigma_video)) / 2.0

            # ── 诊断统计（不参与 loss，detach 后计算，每 N 步采集一次） ──
            self._diag_step += 1
            if self._diag_step % self._diag_interval == 0:
                with torch.no_grad():
                    retrieval_stats = self._matrix_gap_stats(
                        weighted_logits, positive_mask=positive_mask
                    )
                    positive_values = weighted_logits.detach()[positive_mask]
                    self._diag_chain = {
                        "pos_mean": retrieval_stats["diag"],
                        "neg_mean": retrieval_stats["off"],
                        "gap": retrieval_stats["gap"],
                        "pos_std": float(
                            positive_values.std(unbiased=False).item()
                        ) if positive_values.numel() else 0.0,
                    }
                    prob_stats = self._matrix_gap_stats(
                        prob_mu_logits, positive_mask=positive_mask
                    )
                    anchor_stats = self._matrix_gap_stats(
                        anchor_wti_logits, positive_mask=positive_mask
                    )

                    # 链二：Evidential 不确定性统计
                    ls_t = prob_text_logsigma.detach()
                    self._prob_chain = {
                        "u_mode_mean":        float(u_mode.detach().mean()),
                        "u_mode_std":         float(u_mode.detach().std(unbiased=False)),
                        "epistemic_v_mean":   float(epistemic_video.detach().mean()),
                        "var_text_mean":      float(ls_t.exp().mean()),
                        "kl_text_mean":       float((0.5 * (ls_t.exp() - 1 - ls_t)).mean(dim=-1).mean()),
                    }

                    # 链三：辅助 loss 子项
                    self._aux_chain = {
                        "evidential_loss_val":      float(evidential_loss.item()),
                        "neg_reg_loss_val":         float(neg_reg_loss.item()),
                        "logsigma_v_mean":          float(logsigma_video.detach().mean().item()),
                        "logsigma_v_min_ratio":     self._bound_ratio(logsigma_video.detach(), self.log_sigma_min, "min"),
                        "logsigma_v_max_ratio":     self._bound_ratio(logsigma_video.detach(), self.log_sigma_max, "max"),
                        "logsigma_t_min_ratio":     self._bound_ratio(prob_text_logsigma.detach(), self.log_sigma_min, "min"),
                        "logsigma_t_max_ratio":     self._bound_ratio(prob_text_logsigma.detach(), self.log_sigma_max, "max"),
                        "anneal_factor":            float(anneal_factor),
                        "uncertainty_mode":         self.uncertainty_mode,
                        "experiment_profile":       self.experiment_profile,
                        "score_source":             score_source,
                        "final_score_mode":         self.final_score_mode,
                        "lambda_prob":              float(self.lambda_prob),
                        "lambda_anchor":            float(self.lambda_anchor),
                        "lambda_qc_sap":            float(self.lambda_qc_sap),
                        "qc_sap_temperature":       float(self.qc_sap_temperature),
                        "prob_mu_diag":             prob_stats["diag"],
                        "prob_mu_off":              prob_stats["off"],
                        "prob_mu_gap":              prob_stats["gap"],
                        "anchor_wti_diag":          anchor_stats["diag"],
                        "anchor_wti_off":           anchor_stats["off"],
                        "anchor_wti_gap":           anchor_stats["gap"],
                        "qc_sap_diag":              qc_sap_stats.get("diag", 0.0),
                        "qc_sap_off":               qc_sap_stats.get("off", 0.0),
                        "qc_sap_gap":               qc_sap_stats.get("gap", 0.0),
                        "qc_sap_std":               qc_sap_stats.get("std", 0.0),
                        "qc_gate_entropy_pos":      qc_sap_stats.get("gate_entropy_pos", 0.0),
                        "qc_gate_entropy_neg":      qc_sap_stats.get("gate_entropy_neg", 0.0),
                        "qc_gate_top1_mass_pos":    qc_sap_stats.get("gate_top1_mass_pos", 0.0),
                        "qc_gate_top1_mass_neg":    qc_sap_stats.get("gate_top1_mass_neg", 0.0),
                        "active_mil":               int(active["mil"]),
                        "active_evidential":        int(active["evidential"]),
                        "active_neg_reg":           int(active["neg_reg"]),
                        "active_orth":              int(active["orth"]),
                        "active_hard_negative":     int(active["hard_negative"]),
                        "active_uacl":              int(active["uacl_intra"] or active["uacl_kl"]),
                        "hard_negative_loss_val":    float(hard_negative_loss.detach().item()),
                        "hard_diag_mean":            hard_diag_mean if hard_diag_mean is not None else 0.0,
                        "hard_pos_gap":              hard_pos_gap if hard_pos_gap is not None else 0.0,
                        "uacl_intra_loss_val":       float(uacl_intra_loss.detach().item()),
                        "uacl_text_loss_val":        float(uacl_text_loss.detach().item()),
                        "uacl_video_loss_val":       float(uacl_video_loss.detach().item()),
                        "uacl_kl_loss_val":          float(uacl_kl_loss.detach().item()),
                    }
            # ──────────────────────────────────────────────────────────────────

            # Anchor 正交损失：只对 SAP 的 K 个语义 anchor
            orth_loss = torch.tensor(0.0, device=anchors.device)
            if active["orth"]:
                anchor_norm = F.normalize(anchors, dim=-1)
                sim_qq = torch.bmm(anchor_norm, anchor_norm.transpose(1, 2))
                eye = torch.eye(sim_qq.size(1), device=sim_qq.device).unsqueeze(0)
                orth_loss = ((sim_qq - eye) ** 2).mean()

            loss = {}
            loss["retrieve_logits"] = weighted_logits
            loss["video_group_id"] = video_group_id
            loss["MIL_loss"] = self.w_mil * MIL_loss
            # 退火系数应用于 evidential_loss 和 neg_reg_loss，避免 Epoch 2 初期梯度崩塌
            loss["evidential_loss"] = self.w_evidential * evidential_loss * anneal_factor
            loss["neg_reg_loss"] = self.w_neg_reg * neg_reg_loss * anneal_factor
            loss["orth_loss"] = self.w_orth * orth_loss
            loss["hard_negative_loss"] = self.w_hard_negative * hard_negative_loss
            loss["uacl_intra_loss"] = self.w_uacl_intra * uacl_intra_loss
            loss["uacl_kl_loss"] = self.w_uacl_kl * uacl_kl_loss
            return loss
        else:
            return weighted_logits

    @staticmethod
    def _positive_weight(task_config, name, default=0.0):
        return float(getattr(task_config, name, default)) > 0.0

    @staticmethod
    def final_score_uses_sap(task_config):
        return getattr(task_config, "final_score_mode", "wti") in {"wti_prob_mu", "wti_anchor_wti", "wti_qc_sap"}

    @staticmethod
    def final_score_uses_text_probability(task_config):
        return getattr(task_config, "final_score_mode", "wti") == "wti_prob_mu"

    @staticmethod
    def compose_final_retrieval_logits(
        wti_logits,
        final_score_mode="wti",
        lambda_prob=0.0,
        lambda_anchor=0.0,
        lambda_qc_sap=0.0,
        prob_mu_logits=None,
        anchor_wti_logits=None,
        qc_sap_logits=None,
    ):
        if final_score_mode == "wti":
            return wti_logits, "wti_logits"
        if final_score_mode == "wti_prob_mu":
            if prob_mu_logits is None:
                raise ValueError("prob_mu_logits is required when final_score_mode=wti_prob_mu")
            return wti_logits + float(lambda_prob) * prob_mu_logits, "wti_prob_mu"
        if final_score_mode == "wti_anchor_wti":
            if anchor_wti_logits is None:
                raise ValueError("anchor_wti_logits is required when final_score_mode=wti_anchor_wti")
            return wti_logits + float(lambda_anchor) * anchor_wti_logits, "wti_anchor_wti"
        if final_score_mode == "wti_qc_sap":
            if qc_sap_logits is None:
                raise ValueError("qc_sap_logits is required when final_score_mode=wti_qc_sap")
            return wti_logits + float(lambda_qc_sap) * qc_sap_logits, "wti_qc_sap"
        raise ValueError(f"Unsupported final_score_mode={final_score_mode}")

    @staticmethod
    def _matrix_gap_stats(logits, positive_mask=None):
        if logits is None:
            return {"diag": 0.0, "off": 0.0, "gap": 0.0, "std": 0.0}
        logits = logits.detach()
        if positive_mask is not None:
            positive_mask = positive_mask.to(
                device=logits.device, dtype=torch.bool
            )
            if positive_mask.shape != logits.shape:
                raise ValueError(
                    f"positive mask shape={tuple(positive_mask.shape)} "
                    f"does not match logits={tuple(logits.shape)}"
                )
            positive = logits[positive_mask]
            negative = logits[~positive_mask]
        else:
            diag_len = min(logits.size(0), logits.size(1))
            diag_index = torch.arange(diag_len, device=logits.device)
            positive = logits[diag_index, diag_index]
            negative_mask = torch.ones_like(logits, dtype=torch.bool)
            negative_mask[diag_index, diag_index] = False
            negative = logits[negative_mask]
        positive_mean = (
            positive.mean() if positive.numel() else logits.new_zeros(())
        )
        negative_mean = (
            negative.mean() if negative.numel() else logits.new_zeros(())
        )
        return {
            "diag": float(positive_mean.item()),
            "off": float(negative_mean.item()),
            "gap": float((positive_mean - negative_mean).item())
            if positive.numel() and negative.numel()
            else 0.0,
            "std": float(logits.std(unbiased=False).item())
            if logits.numel()
            else 0.0,
        }

    @staticmethod
    def compute_query_conditioned_sap_logits(
        text_pooled,
        anchors,
        logit_scale,
        temperature=0.1,
        positive_mask=None,
    ):
        temperature = max(float(temperature), 1e-6)
        text_norm = F.normalize(text_pooled, dim=-1)
        anchor_norm = F.normalize(anchors, dim=-1)
        gate_scores = torch.einsum("id,jkd->ijk", text_norm, anchor_norm) / temperature
        gate = torch.softmax(gate_scores, dim=-1)
        pair_video = torch.einsum("ijk,jkd->ijd", gate, anchor_norm)
        pair_video = F.normalize(pair_video, dim=-1)
        logits = torch.einsum("id,ijd->ij", text_norm, pair_video) * logit_scale

        stats = UATVR._matrix_gap_stats(logits, positive_mask=positive_mask)
        with torch.no_grad():
            entropy = -(gate.detach() * torch.log(gate.detach() + 1e-8)).sum(dim=-1)
            top1_mass = gate.detach().max(dim=-1).values
            if positive_mask is None:
                diag_len = min(gate.size(0), gate.size(1))
                positive_mask = torch.zeros_like(entropy, dtype=torch.bool)
                diag_idx = torch.arange(diag_len, device=gate.device)
                positive_mask[diag_idx, diag_idx] = True
            else:
                positive_mask = positive_mask.to(
                    device=gate.device, dtype=torch.bool
                )
                if positive_mask.shape != entropy.shape:
                    raise ValueError(
                        f"positive mask shape={tuple(positive_mask.shape)} "
                        f"does not match gate pairs={tuple(entropy.shape)}"
                    )
            pos_entropy = entropy[positive_mask]
            pos_top1 = top1_mass[positive_mask]
            neg_entropy = entropy[~positive_mask]
            neg_top1 = top1_mass[~positive_mask]
            if pos_entropy.numel():
                stats.update(
                    {
                        "gate_entropy_pos": float(pos_entropy.mean().item()),
                        "gate_entropy_neg": float(neg_entropy.mean().item()) if neg_entropy.numel() > 0 else 0.0,
                        "gate_top1_mass_pos": float(pos_top1.mean().item()),
                        "gate_top1_mass_neg": float(neg_top1.mean().item()) if neg_top1.numel() > 0 else 0.0,
                    }
                )
            else:
                stats.update(
                    {
                        "gate_entropy_pos": 0.0,
                        "gate_entropy_neg": 0.0,
                        "gate_top1_mass_pos": 0.0,
                        "gate_top1_mass_neg": 0.0,
                    }
                )
        return logits, stats

    @classmethod
    def should_freeze_parameter_for_profile(cls, name, task_config):
        profile = getattr(task_config, "experiment_profile", "default")
        final_score_mode = getattr(task_config, "final_score_mode", "wti")
        if name.startswith("qc_sap_") and final_score_mode != "wti_qc_sap":
            return True
        if profile != "hygiene":
            return False
        if (
            final_score_mode in {"wti_anchor_wti", "wti_qc_sap"}
            and (name.startswith("sap.anchor_proj.") or name.startswith("sap.evidential_head."))
        ):
            return True
        if name.startswith("sap.") and cls.final_score_uses_sap(task_config):
            return False
        if name.startswith("pie_net_text.") and cls.final_score_uses_text_probability(task_config):
            return False
        if (
            cls.final_score_uses_text_probability(task_config)
            and bool(getattr(task_config, "use_ada_norm", False))
            and (name.startswith("uncertain_net_text.") or name.startswith("ada_norm_text."))
        ):
            return False
        return name.startswith(cls.HYGIENE_FROZEN_PARAMETER_PREFIXES)

    def freeze_inactive_parameters_for_profile(self):
        frozen_count = 0
        for name, param in self.named_parameters():
            if self.should_freeze_parameter_for_profile(name, self.task_config):
                param.requires_grad = False
                frozen_count += 1
        return frozen_count

    @staticmethod
    def resolve_loss_activations(task_config):
        profile = getattr(task_config, "experiment_profile", "default")
        if profile == "hygiene":
            return {
                "mil": False,
                "evidential": False,
                "neg_reg": False,
                "orth": False,
                "hard_negative": False,
                "uacl_intra": False,
                "uacl_kl": False,
            }

        uncertainty_mode = getattr(task_config, "uncertainty_mode", "none")
        use_evidential = uncertainty_mode == "evidential"
        use_uacl = bool(getattr(task_config, "use_uacl_intra_alignment", False))
        use_hn = bool(getattr(task_config, "use_explicit_hard_negative_loss", False))
        return {
            "mil": UATVR._positive_weight(task_config, "w_mil", 1e-2),
            "evidential": use_evidential and UATVR._positive_weight(task_config, "w_evidential", 1e-2),
            "neg_reg": use_evidential and UATVR._positive_weight(task_config, "w_neg_reg", 1e-2),
            "orth": UATVR._positive_weight(task_config, "w_orth", 0.0),
            "hard_negative": use_hn and UATVR._positive_weight(task_config, "w_hard_negative", 5e-2),
            "uacl_intra": use_uacl and UATVR._positive_weight(task_config, "w_uacl_intra", 1e-2),
            "uacl_kl": use_uacl and UATVR._positive_weight(task_config, "w_uacl_kl", 1e-4),
        }

    @staticmethod
    def _bound_ratio(values, bound, mode):
        if bound is None:
            return 0.0
        bound = float(bound)
        if mode == "min":
            return float((values <= bound + 1e-6).to(dtype=torch.float).mean().item())
        if mode == "max":
            return float((values >= bound - 1e-6).to(dtype=torch.float).mean().item())
        raise ValueError(f"Unknown bound ratio mode: {mode}")

    @staticmethod
    def _hard_negative_infonce_loss(retrieval_logits, hard_logits, valid_mask):
        valid_mask = valid_mask.to(device=retrieval_logits.device, dtype=torch.bool).view(-1)
        if valid_mask.numel() == 0 or not bool(valid_mask.any().item()):
            return retrieval_logits.new_tensor(0.0)
        if retrieval_logits.size(0) != retrieval_logits.size(1):
            raise ValueError(f"retrieval_logits must be square, got shape={tuple(retrieval_logits.shape)}")
        if hard_logits.shape != retrieval_logits.shape:
            raise ValueError(
                f"hard_logits shape must match retrieval_logits, got "
                f"{tuple(hard_logits.shape)} vs {tuple(retrieval_logits.shape)}"
            )
        if valid_mask.numel() != hard_logits.size(1):
            raise ValueError(
                f"valid_mask length must match hard-negative columns, got "
                f"{valid_mask.numel()} vs {hard_logits.size(1)}"
            )

        masked_hard = hard_logits.masked_fill(~valid_mask.unsqueeze(0), torch.finfo(hard_logits.dtype).min)
        logits = torch.cat([retrieval_logits, masked_hard], dim=1)
        target = torch.arange(retrieval_logits.size(0), device=retrieval_logits.device)
        return F.cross_entropy(logits, target)

    @staticmethod
    def _select_closest_gaussian_sample(mean, samples):
        mean_norm = F.normalize(mean, dim=-1)
        sample_norm = F.normalize(samples, dim=-1)
        sim = torch.einsum("bd,bnd->bn", mean_norm, sample_norm)
        idx = sim.argmax(dim=1)
        return samples[torch.arange(samples.size(0), device=samples.device), idx]

    @staticmethod
    def _select_uacl_gaussian_sample(mean, samples, strategy="closest"):
        if strategy == "closest":
            return UATVR._select_closest_gaussian_sample(mean, samples)
        if strategy == "random":
            idx = torch.randint(samples.size(1), (samples.size(0),), device=samples.device)
            return samples[torch.arange(samples.size(0), device=samples.device), idx]
        raise ValueError(f"Unknown UACL sample strategy: {strategy}")

    @staticmethod
    def _uacl_intra_contrastive_loss(anchor, positive, temperature=0.07):
        temperature = max(float(temperature), 1e-6)
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        logits = torch.matmul(anchor, positive.t()) / temperature
        target = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, target)

    @staticmethod
    def _logvar_kl(logsigma):
        return (0.5 * (logsigma.exp() - 1.0 - logsigma)).mean()

    @staticmethod
    def _evidential_similarity(mu_video, text_pooled, epistemic_video):
        """Evidential 相似度：cosine sim × 认知不确定性折扣。

        Args:
            mu_video: [B, D] SAP 模态概率聚合均值 (L2 normalized)。
            text_pooled: [B, D] 文本 pooled 表征 (L2 normalized)。
            epistemic_video: [B, K, D] SAP 每锚点认知不确定性 β/(ν(α-1))。

        Returns:
            sim_matrix: [B, B] cosine 相似度 × exp(-mean_epistemic)。
        """
        cosine_sim = torch.mm(mu_video, text_pooled.t())  # [B, B]
        # 锚点维度平均 → 每视频的标量不确定性惩罚
        epistemic_penalty = epistemic_video.mean(dim=(1, 2))  # [B]
        confidence = torch.exp(-epistemic_penalty)  # [B]，不确定性越大，置信越低
        sim_matrix = cosine_sim * confidence.unsqueeze(1)  # [B, B]
        return sim_matrix

    @staticmethod
    def _evidential_nll_loss(sim_matrix, alpha_dir):
        """Evidential 负对数似然损失：鼓励正对高证据、抑制负对证据。

        Args:
            sim_matrix: [B, B] evidential 相似度（非 detach，梯度回传到 SAP 和 text）。
            alpha_dir: [B, K] SAP 的 Dirichlet 证据量。

        Returns:
            loss: 标量，正对 NLL + 负对 evidence 正则。
        """
        B = sim_matrix.size(0)
        diag_scores = sim_matrix.diagonal()  # [B] 正对分数
        # 正对 NLL：relu 截断负值（cosine sim 可能为负），鼓励高证据
        nll_pos = -torch.log(torch.relu(diag_scores) + 1e-8).mean()
        # 负对 evidence 正则：对每行非对角元素求和
        mask_off = ~torch.eye(B, dtype=torch.bool, device=sim_matrix.device)
        neg_scores = sim_matrix[mask_off].view(B, B - 1)  # [B, B-1]
        evidence_neg = torch.relu(neg_scores).sum(dim=-1)  # [B]
        nll_neg = torch.log(1.0 + evidence_neg).mean()
        return nll_pos + nll_neg

    @staticmethod
    def _evidential_neg_reg_loss(sim_matrix):
        """负对 evidence 正则：独立的负对证据量惩罚项。

        Args:
            sim_matrix: [B, B] evidential 相似度（可 detach 也可不 detach）。

        Returns:
            loss: 标量，log(1 + sum(relu(neg))) 的均值。
        """
        B = sim_matrix.size(0)
        mask_off = ~torch.eye(B, dtype=torch.bool, device=sim_matrix.device)
        neg_scores = sim_matrix[mask_off].view(B, B - 1)
        return torch.log(1.0 + torch.relu(neg_scores).sum(dim=-1)).mean()

    @staticmethod
    def evidential_matrix_loss(sim_matrix):
        """DUQ-style evidence loss that encourages diagonal confidence in a similarity matrix.

        Returns:
            (loss, uncertainty): loss 为标量，uncertainty 为 [B] 的 per-sample 不确定性分数。
            uncertainty = 1 - K/S，其中 K=批次大小，S=对角 alpha 之和。
            S 越大（证据越强），uncertainty 越接近 0；S≈K 时 uncertainty≈1。
        """
        B = sim_matrix.shape[0]
        target = torch.eye(B, dtype=sim_matrix.dtype, device=sim_matrix.device)
        evidence = torch.relu(sim_matrix)
        alpha = evidence + 1.0

        def _dirichlet_mse(cur_target, cur_alpha):
            strength = torch.sum(cur_alpha, dim=1, keepdim=True)
            prob = cur_alpha / strength
            err = torch.sum((cur_target - prob) ** 2, dim=1, keepdim=True)
            var = torch.sum(
                cur_alpha * (strength - cur_alpha) / (strength * strength * (strength + 1.0)),
                dim=1,
                keepdim=True,
            )
            return err + var

        loss = (
            torch.mean(_dirichlet_mse(target, alpha))
            + torch.mean(_dirichlet_mse(target, alpha.t()))
        ) / 2.0

        # uncertainty: 对角 alpha 求和得到 S，uncertainty = 1 - K/S
        diag_alpha = alpha.diagonal()  # [B]
        S = diag_alpha.sum()
        uncertainty = 1.0 - B / (S + 1e-8)

        return loss, uncertainty

    def weighted_token_wise_intersection(self, text_token, frame_token, attention_mask, video_mask):
        device = text_token.device
        text_weight = self.text_weight_fc(text_token).squeeze(2)  # B x N_t x D -> B x N_t
        # 构造与输入同设备/同形状的布尔mask
        text_mask_bool = (attention_mask == 0).to(device=device, dtype=torch.bool)
        text_weight.masked_fill_(text_mask_bool, float("-inf"))
        text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t

        video_weight = self.video_weight_fc(frame_token).squeeze(2)  # B x N_v x D -> B x N_v
        video_mask_bool = (video_mask == 0).to(device=device, dtype=torch.bool)
        video_weight.masked_fill_(video_mask_bool, float("-inf"))
        video_weight = torch.softmax(video_weight, dim=-1)  # B x N_v

        # token-wise interaction
        retrieve_logits = torch.einsum("atd,bvd->abtv", [text_token, frame_token])
        retrieve_logits = torch.einsum("abtv,at->abtv", [retrieve_logits, attention_mask])
        retrieve_logits = torch.einsum("abtv,bv->abtv", [retrieve_logits, video_mask])

        t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
        t2v_logits = torch.einsum("abt,at->ab", [t2v_logits, text_weight])

        v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
        v2t_logits = torch.einsum("abv,bv->ab", [v2t_logits, video_weight])
        retrieve_logits = (t2v_logits + v2t_logits) / 2.0
        return retrieve_logits

    def probabilistic_text(self, text_pooled, text_token, attention_mask=None, sample_embeddings=True):
        output = {}
        pad_mask = None
        if attention_mask is not None:
            pad_mask = attention_mask == 0

        out, attn, residual = self.pie_net_text(
            text_pooled, text_token, pad_mask=pad_mask
        )  # (B 512) (B 32 512)   multiheadatt + fc + sigmoid + (residual) + laynorm
        output["attention"] = attn
        output["residual"] = residual

        uncertain_out = self.uncertain_net_text(
            text_pooled, text_token, pad_mask=pad_mask
        )  # (B 512) (B 32 512)   multiheadatt + fc + (residual)
        logsigma = uncertain_out["logsigma"]
        if self.log_sigma_min is not None and self.log_sigma_max is not None:
            logsigma = torch.clamp(logsigma, min=float(self.log_sigma_min), max=float(self.log_sigma_max))
        output["logsigma"] = logsigma
        output["uncertainty_attention"] = uncertain_out["attention"]

        if self.use_ada_norm:
            out = self.ada_norm_text(out, logsigma)
        out = l2_normalize(out)
        output["mean"] = out

        if sample_embeddings:
            output["embedding"] = sample_gaussian_tensors(out, logsigma, self.n_text_samples)

        return output

    def get_similarity_logits(
        self,
        sequence_output,
        text_token,
        visual_output,
        visual_output_hidden,
        attention_mask,
        video_mask,
        shaped=False,
        loose_type=False,
        video_group_id=None,
        hard_visual_output=None,
        hard_video_mask=None,
        hard_valid=None,
    ):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        if self.training:
            loss = self._loose_similarity(
                sequence_output,
                text_token,
                visual_output,
                visual_output_hidden,
                attention_mask,
                video_mask,
                sim_header=self.sim_header,
                video_group_id=video_group_id,
                hard_visual_output=hard_visual_output,
                hard_video_mask=hard_video_mask,
                hard_valid=hard_valid,
            )
            return loss
        else:
            retrieve_logits = self._loose_similarity(
                sequence_output,
                text_token,
                visual_output,
                visual_output_hidden,
                attention_mask,
                video_mask,
                sim_header=self.sim_header,
            )
            return retrieve_logits, {}
