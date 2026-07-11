from __future__ import absolute_import, division, print_function

import logging

import torch
import torch.nn.functional as F
from torch import nn

from modules.backbone_adapter import (
    build_eva_clip_backbone,
    get_eva_clip_backbone_spec,
    load_eva_clip_pretrained,
)
from modules.module_clip import (
    CLIP,
    convert_weights,
    set_layer_norm_precision,
    validate_layer_norm_precision,
)
from modules.module_cross import CrossConfig
from modules.module_cross import Transformer as TransformerClip
from modules.until_module import (
    MultiPositiveCrossEn,
    PreTrainedModel,
    allgather_no_grad,
    allgather_with_grad,
)

logger = logging.getLogger(__name__)


class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    RETIRED_CHECKPOINT_PREFIXES = (
        "sap.",
        "qc_sap_text_proj.",
        "qc_sap_anchor_proj.",
        "pie_net_text.",
        "uncertain_net_text.",
        "ada_norm_text.",
        "spatial_enhancer.",
    )
    RETIRED_CHECKPOINT_ROOT_KEYS = frozenset({"expansion_tokens"})

    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def _validate_retired_checkpoint_keys(cls, state_dict):
        matched = []
        for raw_key in state_dict:
            key = raw_key[7:] if raw_key.startswith("module.") else raw_key
            if key in cls.RETIRED_CHECKPOINT_ROOT_KEYS or key.startswith(
                cls.RETIRED_CHECKPOINT_PREFIXES
            ):
                matched.append(raw_key)
        if matched:
            preview = sorted(matched)[:8]
            raise ValueError(
                "retired auxiliary checkpoint is unsupported; "
                f"matched_keys={preview}, total={len(matched)}. "
                "No automatic migration is provided; use the matching Git revision."
            )

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
        cls._validate_retired_checkpoint_keys(state_dict)
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
    @staticmethod
    def _validate_backbone_contract(
        backbone,
        embed_dim,
        d_model,
        require_hidden=True,
        require_text_hidden=None,
        require_visual_hidden=None,
    ):
        output_dim = getattr(backbone, "output_dim", None)
        dimensions = f"output_dim={output_dim}, embed_dim={embed_dim}, d_model={d_model}"
        if output_dim != embed_dim or output_dim != d_model:
            raise ValueError(f"Backbone dimension contract mismatch: {dimensions}")

        if require_text_hidden is None:
            require_text_hidden = require_hidden
        if require_visual_hidden is None:
            require_visual_hidden = require_hidden
        for capability, required in (
            ("supports_text_hidden", require_text_hidden),
            ("supports_visual_hidden", require_visual_hidden),
        ):
            if required and not getattr(backbone, capability, False):
                raise ValueError(
                    f"Backbone capability contract mismatch: {capability}=False; "
                    "the current UATVR path requires projected token/patch hidden states."
                )

    @staticmethod
    def _validate_eva_spec_capabilities(spec):
        if not getattr(spec, "supports_text_hidden", False):
            raise ValueError("EVA backbone must support text hidden tokens")
        return True, False

    @staticmethod
    def _resolve_hard_negative_enabled(use_explicit, weight):
        weight = float(weight)
        if weight < 0:
            raise ValueError("w_hard_negative must be non-negative")
        return bool(use_explicit) and weight > 0

    @staticmethod
    def configure_clip_layer_norm_precision(backbone, backbone_type, precision):
        precision = validate_layer_norm_precision(precision)
        if backbone_type != "openai_clip":
            return 0
        return set_layer_norm_precision(backbone, precision)

    def __init__(self, cross_config, clip_state_dict, task_config):
        super(UATVR, self).__init__(cross_config)
        self.task_config = task_config

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self.loose_type = False
        if check_attr("loose_type", self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        self.backbone_type = getattr(task_config, "backbone_type", "openai_clip")
        self.clip_layer_norm_precision = validate_layer_norm_precision(
            getattr(task_config, "clip_layer_norm_precision", "fp16")
        )
        self.clip_layer_norm_module_count = 0
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
            require_text_hidden, require_visual_hidden = (
                self._validate_eva_spec_capabilities(spec)
            )
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
            self.clip_layer_norm_module_count = self.configure_clip_layer_norm_precision(
                self.clip,
                backbone_type=self.backbone_type,
                precision=self.clip_layer_norm_precision,
            )
            show_log(
                task_config,
                "\t OpenAI CLIP LayerNorm precision: {} (modules={})".format(
                    self.clip_layer_norm_precision,
                    self.clip_layer_norm_module_count,
                ),
            )
            # OpenAI CLIP's ViT implementation exposes both pooled and token
            # hidden states through encode_text/encode_image(return_hidden=True).
            self.clip.supports_text_hidden = True
            self.clip.supports_visual_hidden = True
            # <=== End of CLIP Encoders
        else:
            self.clip = build_eva_clip_backbone(
                backbone_name=getattr(task_config, "backbone_name", "EVA02-CLIP-B-16"),
                backbone_path=getattr(task_config, "backbone_path", None),
                eva_clip_root=getattr(task_config, "eva_clip_root", None),
                use_xattn=getattr(task_config, "eva_clip_use_xattn", False),
                load_pretrained=False,
            ).float()
            self._validate_backbone_contract(
                self.clip,
                embed_dim=embed_dim,
                d_model=transformer_width,
                require_text_hidden=require_text_hidden,
                require_visual_hidden=require_visual_hidden,
            )
            show_log(
                task_config,
                "\t OpenAI CLIP LayerNorm precision: {} (not applied to EVA)".format(
                    self.clip_layer_norm_precision
                ),
            )

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

        # Token-importance heads for WTI matching
        self.text_weight_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True), nn.Linear(transformer_width, 1)
        )
        self.video_weight_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True), nn.Linear(transformer_width, 1)
        )
        self.loss_fct = MultiPositiveCrossEn()
        self.use_explicit_hard_negative_loss = bool(
            getattr(self.task_config, "use_explicit_hard_negative_loss", False)
        )
        self.w_hard_negative = float(
            getattr(self.task_config, "w_hard_negative", 5e-2)
        )
        self.hard_negative_enabled = self._resolve_hard_negative_enabled(
            self.use_explicit_hard_negative_loss, self.w_hard_negative
        )
        self.experiment_profile = getattr(
            self.task_config, "experiment_profile", "default"
        )

        # extra class token num
        self.extra_cls_frame_num = self.task_config.extra_video_cls_num
        self.extra_cls_text_num = self.task_config.extra_text_cls_num

        show_log(task_config, "CLIP UATVR Model ......")
        show_log(task_config, "\t Extra video Class token number: {}".format(self.extra_cls_frame_num))
        show_log(task_config, "\t Extra text Class token number: {}".format(self.extra_cls_text_num))

        # 诊断统计步数计数器（每 N 步采集一次，减少显存和计算开销）
        self._diag_step = 0
        self._diag_interval = getattr(task_config, "diag_interval", 10)
        self._diag_chain = {}
        self._hard_negative_chain = {}

        self.apply(self.init_weights)

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

        visual_cls = self.get_visual_output(
            video, video_mask, shaped=True, video_frame=video_len
        )

        if self.training:
            hard_visual_cls = None
            if (
                self.hard_negative_enabled
                and hard_video is not None
                and hard_video_mask is not None
            ):
                hard_video_mask = hard_video_mask.view(-1, hard_video_mask.shape[-1])
                hard_video, hard_video_len = self._flatten_video_input(hard_video)
                hard_visual_cls = self.get_visual_output(
                    hard_video,
                    hard_video_mask,
                    shaped=True,
                    video_frame=hard_video_len,
                )

            res = self.get_similarity_logits(
                sequence_output,
                text_token,
                visual_cls,
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
            hard_negative_loss = res["hard_negative_loss"]
            total_loss = sim_loss + hard_negative_loss

            positive_mask = global_group_ids[:, None].eq(
                global_group_ids[None, :]
            )
            positive_counts = positive_mask.sum(dim=1).float()
            unique_count = torch.unique(global_group_ids).numel()
            return {
                "total": total_loss,
                "sim_loss": sim_loss,
                "hard_negative_loss": hard_negative_loss,
                "unique_video_count": sim_matrix.new_tensor(float(unique_count)),
                "duplicate_sample_count": sim_matrix.new_tensor(
                    float(global_group_ids.numel() - unique_count)
                ),
                "mean_positive_count": positive_counts.mean(),
            }
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

    def get_visual_output(
        self, video, video_mask, shaped=False, video_frame=-1
    ):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if video.dim() != 7:
                raise ValueError(
                    f"Expected 7D video tensor, got shape={tuple(video.shape)}"
                )
            b, pair, num_frames, clips_per_frame, channel, h, w = video.shape
            video = video.view(b * pair * num_frames * clips_per_frame, channel, h, w)
            video_frame = num_frames * clips_per_frame

        bs_pair = video_mask.size(0)
        encoded = self.clip.encode_image(
            video, return_hidden=False, video_frame=video_frame
        )
        visual_cls = encoded.float()
        visual_cls = visual_cls.view(bs_pair, -1, visual_cls.size(-1))
        return visual_cls

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
        visual_output = self.get_visual_output(
            video, video_mask, shaped=True, video_frame=video_frame
        )
        return sequence_output, hidden_word, visual_output

    def _wti_similarity(
        self,
        text_token,
        visual_output,
        attention_mask,
        video_mask,
        video_group_id=None,
        hard_visual_output=None,
        hard_video_mask=None,
        hard_valid=None,
    ):
        if self.training:
            visual_output = allgather_with_grad(
                visual_output.contiguous(), self.task_config
            )
            video_mask = allgather_no_grad(
                video_mask.contiguous(), self.task_config
            )
            text_token = allgather_with_grad(
                text_token.contiguous(), self.task_config
            )
            attention_mask = allgather_no_grad(
                attention_mask.contiguous(), self.task_config
            )
            if video_group_id is None:
                raise ValueError("WTI training requires video_group_id")
            video_group_id = allgather_no_grad(
                video_group_id.contiguous().view(-1), self.task_config
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
                        hard_valid.contiguous().view(-1), self.task_config
                    )

        text_token = F.normalize(text_token, dim=-1)
        visual_output = F.normalize(visual_output, dim=-1)
        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = self.weighted_token_wise_intersection(
            text_token, visual_output, attention_mask, video_mask
        ) * logit_scale

        if not self.training:
            return retrieve_logits

        hard_negative_loss = retrieve_logits.new_zeros(())
        hard_diag_mean = 0.0
        hard_pos_gap = 0.0
        if (
            self.hard_negative_enabled
            and hard_visual_output is not None
            and hard_video_mask is not None
        ):
            hard_visual_output = F.normalize(hard_visual_output, dim=-1)
            hard_logits = self.weighted_token_wise_intersection(
                text_token,
                hard_visual_output,
                attention_mask,
                hard_video_mask,
            ) * logit_scale
            if hard_valid is None:
                hard_valid = torch.ones(
                    hard_logits.size(1),
                    dtype=torch.bool,
                    device=hard_logits.device,
                )
            hard_negative_loss = self._hard_negative_infonce_loss(
                retrieve_logits, hard_logits, hard_valid
            )
            valid = hard_valid.to(device=hard_logits.device, dtype=torch.bool)
            if bool(valid.any().item()):
                hard_diag = hard_logits.diagonal()[valid]
                positive_diag = retrieve_logits.diagonal()[valid]
                hard_diag_mean = float(hard_diag.detach().mean().item())
                hard_pos_gap = float(
                    (positive_diag - hard_diag).detach().mean().item()
                )

        positive_mask = video_group_id[:, None].eq(video_group_id[None, :])
        self._diag_step += 1
        if self._diag_step % self._diag_interval == 0:
            stats = self._matrix_gap_stats(
                retrieve_logits, positive_mask=positive_mask
            )
            positive_values = retrieve_logits.detach()[positive_mask]
            self._diag_chain = {
                "pos_mean": stats["diag"],
                "neg_mean": stats["off"],
                "gap": stats["gap"],
                "pos_std": float(
                    positive_values.std(unbiased=False).item()
                ) if positive_values.numel() else 0.0,
            }
            self._hard_negative_chain = {
                "active": int(self.hard_negative_enabled),
                "loss": float(hard_negative_loss.detach().item()),
                "hard_diag_mean": hard_diag_mean,
                "hard_pos_gap": hard_pos_gap,
            }

        return {
            "retrieve_logits": retrieve_logits,
            "video_group_id": video_group_id,
            "hard_negative_loss": self.w_hard_negative * hard_negative_loss,
        }

    def _loose_similarity(
        self,
        sequence_output,
        text_token,
        visual_output,
        attention_mask,
        video_mask,
        sim_header="seqTransf",
        video_group_id=None,
        hard_visual_output=None,
        hard_video_mask=None,
        hard_valid=None,
    ):
        visual_output = visual_output.contiguous()

        if sim_header == "seqTransf":
            visual_output_original = visual_output

            extra_token_num = self.extra_cls_frame_num
            seq_length = visual_output.size(1) + extra_token_num
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=visual_output.device
            )
            position_ids = position_ids.unsqueeze(0).expand(
                visual_output.size(0), -1
            )
            frame_position_embeddings = self.frame_position_embeddings(
                position_ids
            )
            frame_position_embeddings[
                :, 0 : visual_output.size(1), :
            ] += visual_output
            visual_output = frame_position_embeddings

            tempo_mask = torch.cat(
                [
                    video_mask,
                    torch.ones(
                        visual_output.size(0), extra_token_num
                    ).to(visual_output.device),
                ],
                dim=1,
            )
            extended_video_mask = (
                1.0 - tempo_mask.unsqueeze(1)
            ) * -1000000.0
            extended_video_mask = extended_video_mask.expand(
                -1, tempo_mask.size(1), -1
            )
            visual_output = visual_output.permute(1, 0, 2)
            visual_output = self.transformerClip(
                visual_output, extended_video_mask
            )
            visual_output = visual_output.permute(
                1, 0, 2
            ).contiguous()
            visual_output[
                :, : visual_output_original.size(1), :
            ] += visual_output_original
            video_mask = tempo_mask

            text_original = text_token
            extra_text_num = self.extra_cls_text_num
            seq_text_length = extra_text_num + text_token.size(1)
            position_ids_text = torch.arange(
                seq_text_length, dtype=torch.long, device=text_token.device
            )
            position_ids_text = position_ids_text.unsqueeze(0).expand(
                text_token.size(0), -1
            )
            word_position_embeddings = self.word_position_embeddings(
                position_ids_text
            )
            word_position_embeddings[
                :, 0 : text_token.size(1), :
            ] += text_token
            text_token = word_position_embeddings

            tempo_mask_ = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        text_token.size(0), extra_text_num
                    ).to(text_token.device),
                ],
                dim=1,
            )
            extended_text_mask = (
                1.0 - tempo_mask_.unsqueeze(1)
            ) * -1000000.0
            extended_text_mask = extended_text_mask.expand(
                -1, tempo_mask_.size(1), -1
            )
            text_token = text_token.permute(1, 0, 2)
            text_token = self.transformerClip(
                text_token, extended_text_mask
            )
            text_token = text_token.permute(1, 0, 2).contiguous()
            text_token[:, : text_original.size(1), :] += text_original
            attention_mask = tempo_mask_

        return self._wti_similarity(
            text_token,
            visual_output,
            attention_mask,
            video_mask,
            video_group_id=video_group_id,
            hard_visual_output=hard_visual_output,
            hard_video_mask=hard_video_mask,
            hard_valid=hard_valid,
        )

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


    def weighted_token_wise_intersection(
        self, text_token, frame_token, attention_mask, video_mask
    ):
        if text_token.dim() != 3:
            raise ValueError(
                f"text_token must be 3D [A,T,D], got shape={tuple(text_token.shape)}"
            )
        if frame_token.dim() != 3:
            raise ValueError(
                f"frame_token must be 3D [B,V,D], got shape={tuple(frame_token.shape)}"
            )
        if text_token.size(-1) != frame_token.size(-1):
            raise ValueError(
                "WTI feature dimensions must match: "
                f"text={text_token.size(-1)} video={frame_token.size(-1)}"
            )
        if text_token.device != frame_token.device:
            raise ValueError(
                "WTI token devices must match: "
                f"text_token device={text_token.device} "
                f"frame_token device={frame_token.device}"
            )
        if text_token.dtype != frame_token.dtype:
            raise ValueError(
                "WTI token dtypes must match: "
                f"text_token dtype={text_token.dtype} "
                f"frame_token dtype={frame_token.dtype}"
            )
        if not text_token.is_floating_point():
            raise ValueError(
                f"WTI tokens must use a floating dtype, got {text_token.dtype}"
            )

        expected_text_mask_shape = text_token.shape[:2]
        expected_video_mask_shape = frame_token.shape[:2]
        if attention_mask.shape != expected_text_mask_shape:
            raise ValueError(
                f"attention_mask shape={tuple(attention_mask.shape)} "
                f"expected={tuple(expected_text_mask_shape)}"
            )
        if video_mask.shape != expected_video_mask_shape:
            raise ValueError(
                f"video_mask shape={tuple(video_mask.shape)} "
                f"expected={tuple(expected_video_mask_shape)}"
            )
        if attention_mask.device != text_token.device:
            raise ValueError(
                f"attention_mask device={attention_mask.device} does not match "
                f"text_token device={text_token.device}"
            )
        if video_mask.device != frame_token.device:
            raise ValueError(
                f"video_mask device={video_mask.device} does not match "
                f"frame_token device={frame_token.device}"
            )
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all().item()):
            raise ValueError("attention_mask must be binary with values 0 or 1")
        if not bool(((video_mask == 0) | (video_mask == 1)).all().item()):
            raise ValueError("video_mask must be binary with values 0 or 1")

        text_valid = attention_mask.to(dtype=torch.bool)
        video_valid = video_mask.to(dtype=torch.bool)
        empty_text = (~text_valid.any(dim=1)).nonzero(as_tuple=False).view(-1)
        empty_video = (~video_valid.any(dim=1)).nonzero(as_tuple=False).view(-1)
        if empty_text.numel():
            raise ValueError(
                f"WTI no valid text token at batch indices={empty_text.tolist()}"
            )
        if empty_video.numel():
            raise ValueError(
                f"WTI no valid video frame at batch indices={empty_video.tolist()}"
            )

        text_weight = self.text_weight_fc(text_token).squeeze(-1)
        text_weight = text_weight.masked_fill(~text_valid, float("-inf"))
        text_weight = torch.softmax(text_weight, dim=-1)
        video_weight = self.video_weight_fc(frame_token).squeeze(-1)
        video_weight = video_weight.masked_fill(~video_valid, float("-inf"))
        video_weight = torch.softmax(video_weight, dim=-1)

        similarities = torch.einsum("atd,bvd->abtv", text_token, frame_token)
        pair_valid = (
            text_valid[:, None, :, None] & video_valid[None, :, None, :]
        )
        similarities = similarities.masked_fill(
            ~pair_valid, torch.finfo(similarities.dtype).min
        )
        t2v_logits = similarities.max(dim=-1).values
        t2v_logits = t2v_logits.masked_fill(~text_valid[:, None, :], 0.0)
        t2v_logits = torch.einsum("abt,at->ab", t2v_logits, text_weight)
        v2t_logits = similarities.max(dim=-2).values
        v2t_logits = v2t_logits.masked_fill(~video_valid[None, :, :], 0.0)
        v2t_logits = torch.einsum("abv,bv->ab", v2t_logits, video_weight)
        return (t2v_logits + v2t_logits) / 2.0


    def get_similarity_logits(
        self,
        sequence_output,
        text_token,
        visual_output,
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
                attention_mask,
                video_mask,
                sim_header=self.sim_header,
            )
            return retrieve_logits, {}
