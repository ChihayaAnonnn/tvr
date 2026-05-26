from __future__ import absolute_import, division, print_function

import logging
import math

import torch
import torch.nn.functional as F
from torch import nn

from modules.module_clip import CLIP, convert_weights
from modules.module_cross import CrossConfig
from modules.module_cross import Transformer as TransformerClip
from modules.spatial_enhancer import SpatialEnhancer
from modules.until_module import AllGather, CrossEn, KLdivergence, MILNCELoss_BoF, PreTrainedModel
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
allgather = AllGather.apply


class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

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
        if hasattr(task_config, "pretrained_clip_name"):
            pretrained_clip_name = task_config.pretrained_clip_name
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(
            cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config
        )

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)  # -----------

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
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

        if model.sim_header == "seqTransf":
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
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(UATVR, self).__init__(cross_config)
        self.task_config = task_config

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self.loose_type = False
        if check_attr("loose_type", self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

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
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith("transformer.resblocks")))

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

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
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

        self.pie_net_video = PIENet(1, embed_dim, embed_dim, embed_dim // 2)
        # Video-side uncertainty now comes from SAP per-anchor decomposition.
        # self.uncertain_net_video = UncertaintyModuleImage(embed_dim, embed_dim, embed_dim // 2)

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
            self.ada_norm_video = UncertaintyAdaNorm(embed_dim)
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
        self.loss_fct = CrossEn()
        self.loss_MIL_fct = MILNCELoss_BoF()
        self.vib_loss = KLdivergence()

        # Loss weights
        self.w_mil = getattr(self.task_config, "w_mil", 1e-2)
        self.w_vib = getattr(self.task_config, "w_vib", 1e-4)
        self.w_uncertainty_reg = getattr(self.task_config, "w_uncertainty_reg", 1e-3)
        self.w_orth = getattr(self.task_config, "w_orth", 0.0)
        self.use_tas_uncertainty = getattr(self.task_config, "use_tas_uncertainty", False)
        self.tas_top_k = getattr(self.task_config, "tas_top_k", 5)
        self.tas_temperature = getattr(self.task_config, "tas_temperature", 0.05)
        self.tas_kl_scale = getattr(self.task_config, "tas_kl_scale", 0.5)

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

        # 诊断统计步数计数器（每 N 步采集一次，减少显存和计算开销）
        self._diag_step = 0
        self._diag_interval = getattr(task_config, "diag_interval", 10)
        self._diag_chain = {}
        self._prob_chain = {}
        self._aux_chain = {}

        self.apply(self.init_weights)

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        video,
        video_mask=None,
    ):
        # (B 1 32)  (B 1 132) (B 1 32)
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        # video_mask: [B, pair, num_frames] -> [B*pair, num_frames]
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        # Expected video shape from dataloader:
        #   [B, pair, num_frames, clips_per_frame(=1), 3, H, W]
        assert video.dim() == 7, f"Expected 7D video tensor, got shape={tuple(video.shape)}"
        b, pair, num_frames, clips_per_frame, channel, h, w = video.shape
        video = video.view(b * pair * num_frames * clips_per_frame, channel, h, w)
        video_len = num_frames * clips_per_frame

        sequence_output, text_token = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)

        visual_cls, visual_hidden = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_len)

        if self.training:
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
            )
            sim_matrix = res["retrieve_logits"]
            sim_loss1 = self.loss_fct(sim_matrix)
            sim_loss2 = self.loss_fct(sim_matrix.T)
            sim_loss = (sim_loss1 + sim_loss2) / 2

            loss += sim_loss
            loss += res["MIL_loss"]
            loss += res["vib_loss"]
            loss += res["uncertainty_reg_loss"]
            loss += res["orth_loss"]

            loss_dict = {
                "total": loss,
                "sim_loss": sim_loss,
                "mcsoft_loss": res["MIL_loss"],
                "vib_loss": res["vib_loss"],
                "uncertainty_reg": res["uncertainty_reg_loss"],
                "orth_loss": res["orth_loss"],
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
    ):
        sequence_output, visual_output = (
            sequence_output.contiguous(),
            visual_output.contiguous(),
        )  # visual_output [B, T, D]
        visual_output_hidden = visual_output_hidden.contiguous()  # [B, T, 197, D] (ViT-B/16: 196 patches + 1 CLS)
        frame_num = visual_output.size(1)  # 12 / 64
        word_num = text_token.size(1)  # 32 /

        if sim_header == "seqTransf":
            # Sequential type: Transformer Encoder +++++++++++++= extra token
            visual_output_original = visual_output

            extra_token_num = self.extra_cls_frame_num  # extra 2
            seq_length = visual_output.size(1) + extra_token_num  # extra 2 learnable token
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)  # bs num+extra_token_num dim
            frame_position_embeddings[:, 0 : visual_output.size(1), :] += visual_output
            visual_output = frame_position_embeddings

            tempo_mask = torch.cat(
                [video_mask, torch.ones(visual_output.size(0), extra_token_num).to(visual_output.device)], axis=1
            )
            extended_video_mask = (1.0 - tempo_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, tempo_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2).contiguous()  # LND -> NLD
            # multi extra token
            # visual_output = visual_output[:, :visual_output_original.size(1), :] + visual_output_original   # residual fusion
            visual_output[:, : visual_output_original.size(1), :] += visual_output_original
            video_mask = tempo_mask

            # sequential type: MLP for text with extra token
            text_original = text_token  # save original
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

        if self.training:  # DDP all_gather
            visual_output = allgather(visual_output, self.task_config)
            visual_output_hidden = allgather(visual_output_hidden, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            text_token = allgather(text_token, self.task_config)
            attention_mask = allgather(attention_mask, self.task_config)

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        text_token = text_token / text_token.norm(dim=-1, keepdim=True)
        text_pooled = self._mean_pooling_for_similarity_sequence(
            text_token[:, 0:word_num, :].contiguous(), attention_mask[:, 0:word_num].contiguous()
        )
        text_pooled = text_pooled / text_pooled.norm(dim=-1, keepdim=True)
        query_weight = None
        if self.training and self.use_tas_uncertainty:
            with torch.no_grad():
                query_weight = self._compute_lightweight_tas(
                    text_pooled.detach(),
                    top_k=self.tas_top_k,
                    temperature=self.tas_temperature,
                )

        # ===== SAP: Semantic Anchor Probing =====
        # 将时空 token [B, T, 197, D] 展平为 [B, T*197, D]，锚点聚焦局部 patch
        B_vis = visual_output_hidden.size(0)
        T_vis = visual_output_hidden.size(1)
        S_vis = visual_output_hidden.size(2)  # 每帧 patch 数（含 CLS）
        spatial_tokens = visual_output_hidden.reshape(B_vis, T_vis * S_vis, -1)
        # 构造 padding mask：原始 video_mask [B, T] → [B, T*S]
        spatial_mask = video_mask[:, 0:frame_num].unsqueeze(-1).expand(-1, -1, S_vis).reshape(B_vis, -1)
        sap_out = self.sap(
            spatial_tokens.contiguous(),
            padding_mask=(spatial_mask == 0),
        )
        anchors = sap_out["anchors"]           # [B, N, D]
        mu_raw = sap_out["mu_raw"]             # [B, D]
        logsigma_video = sap_out["logsigma"]   # [B, D]
        if self.log_sigma_min is not None and self.log_sigma_max is not None:
            logsigma_video = torch.clamp(logsigma_video, min=float(self.log_sigma_min), max=float(self.log_sigma_max))

        # 拼接 expansion tokens 到 anchor 序列
        if self.num_expansion_tokens > 0:
            B = anchors.size(0)
            exp_tokens = self.expansion_tokens.expand(B, -1, -1)  # [B, E, D]
            anchors = torch.cat([anchors, exp_tokens], dim=1)     # [B, N+E, D]

        # ===== Probabilistic modeling (video side, from SAP anchors) =====
        # PIENet: 用 text_pooled 做 cross-modal query 去 attend anchors，
        # 而非用 mu_raw（已是 anchors 的加权聚合，信息增益有限）
        out_v, _attn_v, _res_v = self.pie_net_video(text_pooled, anchors)
        if self.use_ada_norm:
            out_v = self.ada_norm_video(out_v, logsigma_video)
        out_v = l2_normalize(out_v)
        prob_video_embedding = sample_gaussian_tensors(out_v, logsigma_video, self.n_video_samples)
        prob_video_logsigma = logsigma_video

        # ===== Probabilistic modeling (text side, unchanged) =====
        prob_text = self.probabilistic_text(text_pooled, text_token[:, 0:word_num, :].contiguous())
        prob_text_embedding = prob_text["embedding"]
        prob_text_logsigma = prob_text["logsigma"]

        # ===== Retrieval logits: frame WTI =====
        logit_scale = self.clip.logit_scale.exp()
        wti_logits = self.weighted_token_wise_intersection(
            text_token, visual_output, attention_mask, video_mask
        ) * logit_scale

        if self.training:
            bs = prob_video_embedding.size(0)
            n_video = self.n_video_samples
            n_text = self.n_text_samples
            dim = prob_video_embedding.size(-1)

            prob_sim_v = torch.einsum("ad,bd->ab", prob_video_embedding.view(-1, dim), prob_text_embedding.view(-1, dim))
            prob_sim_t = torch.einsum("ad,bd->ab", prob_text_embedding.view(-1, dim), prob_video_embedding.view(-1, dim))
            MIL_loss = (self.loss_MIL_fct(prob_sim_v, bs, n_video, n_text)
                        + self.loss_MIL_fct(prob_sim_t, bs, n_video, n_text)) / 2

            if query_weight is None:
                vib_loss = self.vib_loss(prob_video_embedding, prob_video_logsigma, prob_text_embedding, prob_text_logsigma)
            else:
                vib_loss = self._dynamic_vib_loss(
                    prob_video_embedding,
                    prob_video_logsigma,
                    prob_text_embedding,
                    prob_text_logsigma,
                    query_weight=query_weight,
                    tas_kl_scale=self.tas_kl_scale,
                )

            # detach 避免 evidential loss 与 CrossEn 的梯度方向冲突
            uncertainty_reg_loss, evidential_uncertainty = self.evidential_matrix_loss(wti_logits.detach())

            # ── 因果链诊断统计（不参与 loss，detach 后计算，每 N 步采集一次） ──
            self._diag_step += 1
            if self._diag_step % self._diag_interval == 0:
                with torch.no_grad():
                    # 链一：检索头 — pos/neg gap
                    S = wti_logits.detach()                  # [B, B]
                    diag = S.diagonal()                      # [B]
                    B_s = S.size(0)
                    mask_off = ~torch.eye(B_s, dtype=torch.bool, device=S.device)
                    off = S[mask_off]
                    self._diag_chain = {
                        "pos_mean": float(diag.mean()),
                        "neg_mean": float(off.mean()),
                        "gap":      float(diag.mean() - off.mean()),
                        "pos_std":  float(diag.std(unbiased=False)),
                    }

                    # 链二：概率分支 — logsigma 量级 + per-sample KL
                    ls_v = prob_video_logsigma.detach()       # [B, D]
                    ls_t = prob_text_logsigma.detach()
                    var_v = float(ls_v.exp().mean())
                    var_t = float(ls_t.exp().mean())
                    kl_v  = 0.5 * (ls_v.exp() - 1 - ls_v).mean(dim=-1)   # [B]
                    kl_t  = 0.5 * (ls_t.exp() - 1 - ls_t).mean(dim=-1)
                    self._prob_chain = {
                        "var_video_mean": var_v,
                        "var_text_mean":  var_t,
                        "kl_video_mean":  float(kl_v.mean()),
                        "kl_video_std":   float(kl_v.std(unbiased=False)),
                        "kl_text_mean":   float(kl_t.mean()),
                        "kl_text_std":    float(kl_t.std(unbiased=False)),
                    }
                    if query_weight is not None:
                        self._prob_chain["tas_mean"] = float(query_weight.detach().mean())
                        self._prob_chain["tas_std"] = float(query_weight.detach().std(unbiased=False))

                    # 链三：辅助 loss 子项
                    self._aux_chain = {
                        "uncertainty_reg_loss_val": float(uncertainty_reg_loss.item()),
                        "evidential_uncertainty": float(evidential_uncertainty.item()),
                        "logsigma_v_mean":   float(logsigma_video.detach().mean().item()),
                    }
            # ──────────────────────────────────────────────────────────────────

            # Anchor 正交损失：只对 SAP 的 N 个语义 anchor 做正交，排除 expansion tokens
            orth_loss = torch.tensor(0.0, device=anchors.device)
            if self.w_orth > 0:
                sap_anchors = anchors[:, :self.num_anchors, :]  # [B, N, D]
                anchor_norm = torch.nn.functional.normalize(sap_anchors, dim=-1)
                sim_qq = torch.bmm(anchor_norm, anchor_norm.transpose(1, 2))  # [B, N, N]
                eye = torch.eye(sim_qq.size(1), device=sim_qq.device).unsqueeze(0)
                orth_loss = ((sim_qq - eye) ** 2).mean()

            loss = {}
            loss["retrieve_logits"] = wti_logits
            loss["MIL_loss"] = self.w_mil * MIL_loss
            loss["vib_loss"] = self.w_vib * vib_loss
            loss["uncertainty_reg_loss"] = self.w_uncertainty_reg * uncertainty_reg_loss
            loss["orth_loss"] = self.w_orth * orth_loss
            if query_weight is not None:
                loss["tas_weight"] = query_weight.detach().mean()
            return loss
        else:
            return wti_logits

    @staticmethod
    def _compute_lightweight_tas(text_pooled, top_k=5, temperature=0.05):
        """基于 batch 内文本相似度分布计算轻量 TAS。"""
        batch_size = text_pooled.size(0)
        if batch_size <= 1:
            return text_pooled.new_zeros(batch_size)

        top_k = max(1, min(int(top_k), batch_size))
        temperature = max(float(temperature), 1e-6)
        text_norm = F.normalize(text_pooled, p=2, dim=-1)
        sim_matrix = torch.matmul(text_norm, text_norm.t())
        top_scores = torch.topk(sim_matrix, k=top_k, dim=-1).values
        probs = torch.softmax(top_scores / temperature, dim=-1)
        entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1)
        if top_k <= 1:
            return torch.zeros_like(entropy)
        return torch.clamp(entropy / math.log(top_k), min=0.0, max=1.0)

    @staticmethod
    def _dynamic_vib_loss(sampled_video, video_sigma, sampled_text, text_sigma, query_weight=None, tas_kl_scale=0.5):
        """按 TAS 对文本侧 KL 做逐样本动态缩放。"""
        video_mu = sampled_video.mean(dim=1)
        text_mu = sampled_text.mean(dim=1)
        video_kl = UATVR._kl_divergence_per_sample(video_mu, video_sigma)
        text_kl = UATVR._kl_divergence_per_sample(text_mu, text_sigma)
        if query_weight is not None:
            query_weight = query_weight.to(device=text_kl.device, dtype=text_kl.dtype).view(-1)
            tas_kl_scale = max(float(tas_kl_scale), 0.0)
            text_scale = 1.0 + tas_kl_scale * (1.0 - 2.0 * query_weight)
            text_scale = torch.clamp(text_scale, min=0.0)
            text_kl = text_kl * text_scale
        return video_kl.mean() + text_kl.mean()

    @staticmethod
    def _kl_divergence_per_sample(mu, logsigma):
        """计算每个样本到标准高斯先验的 KL。"""
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=-1)

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

    def probabilistic_text(self, text_pooled, text_token):
        output = {}

        out, attn, residual = self.pie_net_text(
            text_pooled, text_token
        )  # (B 512) (B 32 512)   multiheadatt + fc + sigmoid + (residual) + laynorm
        output["attention"] = attn
        output["residual"] = residual

        uncertain_out = self.uncertain_net_text(
            text_pooled, text_token
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
