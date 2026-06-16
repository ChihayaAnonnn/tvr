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
from modules.until_module import AllGather, CrossEn, MILNCELoss_BoF, PreTrainedModel
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

        # 只加载 config，不需加载 weights（cross 模块随机初始化训练）。
        # 传 state_dict={} 跳过 get_config 中"权重文件不存在"的检查。
        cross_config, _ = CrossConfig.get_config(
            cross_model_name, cache_dir, type_vocab_size, state_dict={}, task_config=task_config
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

        # Loss weights
        self.w_mil = getattr(self.task_config, "w_mil", 1e-2)
        self.w_evidential = getattr(self.task_config, "w_evidential", 1e-2)
        self.w_neg_reg = getattr(self.task_config, "w_neg_reg", 1e-2)
        self.w_uncertainty_reg = getattr(self.task_config, "w_uncertainty_reg", 1e-3)
        self.w_orth = getattr(self.task_config, "w_orth", 0.0)

        # 退火系数：默认关闭；warmup_epochs > 0 时前 N 个 epoch 线性增大到 1.0
        self.anneal_warmup_epochs = getattr(self.task_config, "anneal_warmup_epochs", 0)
        self._current_epoch = 0  # 由外部 train_loop 更新

        # 方案 A：不确定性置信度 warmup（步数级别）
        self._current_step = 0
        self.warmup_steps = getattr(self.task_config, "warmup_steps", 500)

        # per-pair 置信度编码器：从 text-anchor 注意力分布 → 匹配置信度
        # 输入 detach 切断到 text/video encoder 的梯度，仅 ~150 可学习参数
        num_anchors = self.sap.num_anchors
        self.confidence_mlp = nn.Sequential(
            nn.Linear(num_anchors, num_anchors // 2),
            nn.ReLU(),
            nn.Linear(num_anchors // 2, 1),
            nn.Sigmoid(),
        )

        # 不确定性训练模式：nig_mil / none
        self.uncertainty_mode = getattr(self.task_config, "uncertainty_mode", "none")

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
            self._current_step += 1
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
            loss += res["evidential_loss"]
            loss += res["neg_reg_loss"]
            loss += res["orth_loss"]

            loss_dict = {
                "total": loss,
                "sim_loss": sim_loss,
                "mcsoft_loss": res["MIL_loss"],
                "evidential_loss": res["evidential_loss"],
                "neg_reg_loss": res["neg_reg_loss"],
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
        modal_probs = sap_out["modal_probs"]   # [B, K] 模态概率

        if self.log_sigma_min is not None and self.log_sigma_max is not None:
            logsigma_video = torch.clamp(logsigma_video, min=float(self.log_sigma_min), max=float(self.log_sigma_max))

        # ===== DDP allgather：只广播紧凑张量，不再广播 visual_output_hidden =====
        if self.training:
            mu_video = allgather(mu_video.contiguous(), self.task_config)
            logsigma_video = allgather(logsigma_video.contiguous(), self.task_config)
            epistemic_video = allgather(epistemic_video.contiguous(), self.task_config)
            anchors = allgather(anchors.contiguous(), self.task_config)
            u_mode = allgather(u_mode.contiguous(), self.task_config)
            alpha_dir = allgather(alpha_dir.contiguous(), self.task_config)
            visual_output = allgather(visual_output.contiguous(), self.task_config)
            video_mask = allgather(video_mask.contiguous(), self.task_config)
            sequence_output = allgather(sequence_output.contiguous(), self.task_config)
            text_token = allgather(text_token.contiguous(), self.task_config)
            attention_mask = allgather(attention_mask.contiguous(), self.task_config)

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        text_token = text_token / text_token.norm(dim=-1, keepdim=True)
        text_pooled = self._mean_pooling_for_similarity_sequence(
            text_token[:, 0:word_num, :].contiguous(), attention_mask[:, 0:word_num].contiguous()
        )
        text_pooled = text_pooled / text_pooled.norm(dim=-1, keepdim=True)

        # ===== Evidential 相似度：cosine × epistemic 置信折扣 =====
        ev_sim = self._evidential_similarity(mu_video, text_pooled, epistemic_video)

        # ===== Text 侧概率建模（保持 Gaussian，非对称设计） =====
        prob_text = self.probabilistic_text(text_pooled, text_token[:, 0:word_num, :].contiguous())
        prob_text_embedding = prob_text["embedding"]
        prob_text_logsigma = prob_text["logsigma"]

        # ===== WTI：帧级 token 交互打分 =====
        logit_scale = self.clip.logit_scale.exp()
        wti_logits = self.weighted_token_wise_intersection(
            text_token, visual_output, attention_mask, video_mask
        ) * logit_scale

        if self.training:
            bs = mu_video.size(0)
            n_video = self.n_video_samples
            n_text = self.n_text_samples
            dim = mu_video.size(-1)

            # MIL loss：从 N(gamma, logsigma) 采样做多实例对比
            prob_video_embedding = sample_gaussian_tensors(mu_video, logsigma_video, n_video)
            prob_sim_v = torch.einsum("ad,bd->ab", prob_video_embedding.view(-1, dim), prob_text_embedding.view(-1, dim))
            prob_sim_t = torch.einsum("ad,bd->ab", prob_text_embedding.view(-1, dim), prob_video_embedding.view(-1, dim))
            MIL_loss = (self.loss_MIL_fct(prob_sim_v, bs, n_video, n_text)
                        + self.loss_MIL_fct(prob_sim_t, bs, n_video, n_text)) / 2

            # Evidential NLL + neg_reg：仅在 uncertainty_mode=none 时保留（作为对照）
            # 方案 A/B 通过 MIL 或 WTI confidence 训练不确定性，不再需要独立的 evidential loss
            if self.uncertainty_mode == "none":
                evidential_loss = self._evidential_nll_loss(ev_sim, alpha_dir)
                neg_reg_loss = self._evidential_neg_reg_loss(ev_sim)
            else:
                evidential_loss = torch.tensor(0.0, device=mu_video.device)
                neg_reg_loss = torch.tensor(0.0, device=mu_video.device)

            # 退火系数：warmup_epochs <= 0 时关闭退火，直接使用完整权重
            if self.anneal_warmup_epochs <= 0:
                anneal_factor = 1.0
            else:
                anneal_factor = min(1.0, self._current_epoch / self.anneal_warmup_epochs)

            # per-pair 置信度：独立 MLP 从 text-anchor 注意力分布预测匹配置信度
            # 输入全部 detach，梯度只传 MLP（~150 参数），不污染 text/video encoder
            anchors_n = F.normalize(anchors.detach(), dim=-1)              # [B, K, D]
            text_n = F.normalize(text_pooled.detach(), dim=-1)              # [B, D]
            pairwise_sim = torch.einsum('id,jkd->ijk', text_n, anchors_n)  # [B, B, K]
            logit_scale = self.clip.logit_scale.exp()
            anchor_attn = F.softmax(pairwise_sim * logit_scale, dim=-1)    # [B, B, K]
            # 注意力熵（诊断用）
            anchor_ent = -(anchor_attn * torch.log(anchor_attn + 1e-8)).sum(dim=-1)  # [B, B]
            anchor_ent = anchor_ent / math.log(self.sap.num_anchors)        # ∈ [0, 1]
            # 独立的置信度编码器
            confidence = self.confidence_mlp(anchor_attn).squeeze(-1)       # [B, B]
            if self.warmup_steps > 0:
                alpha = min(1.0, self._current_step / self.warmup_steps)
                confidence = 1.0 - alpha + alpha * confidence
            else:
                alpha = 1.0
            weighted_logits = wti_logits * confidence  # [B, B] * [B, B]

            # ── 诊断统计（不参与 loss，detach 后计算，每 N 步采集一次） ──
            self._diag_step += 1
            if self._diag_step % self._diag_interval == 0:
                with torch.no_grad():
                    S = wti_logits.detach()
                    diag = S.diagonal()
                    B_s = S.size(0)
                    mask_off = ~torch.eye(B_s, dtype=torch.bool, device=S.device)
                    off = S[mask_off]
                    self._diag_chain = {
                        "pos_mean": float(diag.mean()),
                        "neg_mean": float(off.mean()),
                        "gap":      float(diag.mean() - off.mean()),
                        "pos_std":  float(diag.std(unbiased=False)),
                    }

                    # 链二：Per-pair 置信度统计
                    ls_t = prob_text_logsigma.detach()
                    conf_val = confidence.detach()  # [B, B] per-pair 置信度
                    conf_diag = conf_val.diagonal()
                    B_c = conf_val.size(0)
                    mask_off_c = ~torch.eye(B_c, dtype=torch.bool, device=conf_val.device)
                    conf_off = conf_val[mask_off_c]
                    self._prob_chain = {
                        "anchor_ent_mean":    float(anchor_ent.detach().mean()),
                        "confidence_mean":    float(conf_val.mean()),
                        "confidence_diag":    float(conf_diag.mean()),
                        "confidence_off":     float(conf_off.mean()),
                        "confidence_gap":     float(conf_diag.mean() - conf_off.mean()),
                        "u_mode_mean":        float(u_mode.detach().mean()),
                        "u_mode_std":         float(u_mode.detach().std(unbiased=False)),
                        "epistemic_v_mean":   float(epistemic_video.detach().mean()),
                        "var_text_mean":      float(ls_t.exp().mean()),
                        "kl_text_mean":       float((0.5 * (ls_t.exp() - 1 - ls_t)).mean(dim=-1).mean()),
                        "warmup_alpha":       float(alpha),
                    }

                    # 链三：辅助 loss 子项
                    self._aux_chain = {
                        "evidential_loss_val":      float(evidential_loss.item()),
                        "neg_reg_loss_val":         float(neg_reg_loss.item()),
                        "logsigma_v_mean":          float(logsigma_video.detach().mean().item()),
                        "anneal_factor":            float(anneal_factor),
                        "uncertainty_mode":         self.uncertainty_mode,
                    }
            # ──────────────────────────────────────────────────────────────────

            # Anchor 正交损失：只对 SAP 的 K 个语义 anchor
            orth_loss = torch.tensor(0.0, device=anchors.device)
            if self.w_orth > 0:
                anchor_norm = F.normalize(anchors, dim=-1)
                sim_qq = torch.bmm(anchor_norm, anchor_norm.transpose(1, 2))
                eye = torch.eye(sim_qq.size(1), device=sim_qq.device).unsqueeze(0)
                orth_loss = ((sim_qq - eye) ** 2).mean()

            loss = {}
            loss["retrieve_logits"] = weighted_logits
            loss["MIL_loss"] = self.w_mil * MIL_loss
            # 退火系数应用于 evidential_loss 和 neg_reg_loss，避免 Epoch 2 初期梯度崩塌
            loss["evidential_loss"] = self.w_evidential * evidential_loss * anneal_factor
            loss["neg_reg_loss"] = self.w_neg_reg * neg_reg_loss * anneal_factor
            loss["orth_loss"] = self.w_orth * orth_loss
            return loss
        else:
            return wti_logits

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
