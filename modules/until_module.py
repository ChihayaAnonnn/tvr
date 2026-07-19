# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

import logging
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.append("..")
from modules.until_config import PretrainedConfig

logger = logging.getLogger(__name__)


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class PreTrainedModel(nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            if "beta" in dir(module) and "gamma" in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def resize_token_embeddings(self, new_num_tokens=None):
        raise NotImplementedError

    @classmethod
    def init_preweight(cls, model, state_dict, prefix=None, task_config=None):
        # Key normalization below is intentionally local: callers may reuse
        # the checkpoint mapping for validation or another model instance.
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        if prefix is not None:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                old_keys.append(key)
                new_keys.append(prefix + key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(model, prefix="")

        if prefix is None and (task_config is None or task_config.local_rank == 0):
            if len(missing_keys) > 0:
                # 按模块分组统计，避免逐行罗列上百个参数
                _mod_cnt = {}
                for k in missing_keys:
                    _mod = k.split(".")[0] if "." in k else "(root)"
                    _mod_cnt[_mod] = _mod_cnt.get(_mod, 0) + 1
                _mod_summary = ", ".join(f"{m}×{c}" for m, c in sorted(_mod_cnt.items()))
                logger.info(
                    "%s: %d 个参数为新增模块，将随机初始化 (%s)",
                    model.__class__.__name__, len(missing_keys), _mod_summary,
                )
            if len(unexpected_keys) > 0:
                logger.info(
                    "%s: %d 个 pretrained 参数在当前模型中未使用（通常为 config 元数据），已忽略",
                    model.__class__.__name__, len(unexpected_keys),
                )
            if len(error_msgs) > 0:
                logger.error(
                    "Weights from pretrained model cause errors in {}: {}".format(
                        model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)
                    )
                )

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )

        return model

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [
                    (k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)
                ]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    @classmethod
    def from_pretrained(cls, config, state_dict=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            return model
        model = cls.init_preweight(model, state_dict)

        return model


##################################
###### LOSS FUNCTION #############
##################################
class CrossEn(nn.Module):
    def __init__(
        self,
    ):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


class MultiPositiveCrossEn(nn.Module):
    _INTEGER_DTYPES = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }

    @classmethod
    def _prepare_group_ids(cls, group_ids, name, device):
        group_ids = torch.as_tensor(group_ids)
        if group_ids.dtype not in cls._INTEGER_DTYPES:
            raise ValueError(
                f"{name} must use an integer dtype, got dtype={group_ids.dtype}"
            )
        return group_ids.reshape(-1).to(device=device, dtype=torch.long)

    def forward(self, logits, query_group_ids, candidate_group_ids=None):
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D, got shape={tuple(logits.shape)}")
        query_group_ids = self._prepare_group_ids(
            query_group_ids, "query_group_ids", logits.device
        )
        if candidate_group_ids is None:
            candidate_group_ids = query_group_ids
        else:
            candidate_group_ids = self._prepare_group_ids(
                candidate_group_ids, "candidate_group_ids", logits.device
            )
        expected_shape = (
            query_group_ids.numel(),
            candidate_group_ids.numel(),
        )
        if tuple(logits.shape) != expected_shape:
            raise ValueError(
                f"logit/group shape mismatch: logits={tuple(logits.shape)} "
                f"query={query_group_ids.numel()} "
                f"candidate={candidate_group_ids.numel()}"
            )

        positive_mask = query_group_ids[:, None].eq(
            candidate_group_ids[None, :]
        )
        missing = (~positive_mask.any(dim=1)).nonzero(as_tuple=False).reshape(-1)
        if missing.numel():
            missing_groups = query_group_ids[missing]
            raise ValueError(
                "multi-positive target missing positive "
                f"query indices={missing.tolist()} "
                f"group IDs={missing_groups.tolist()}"
            )

        positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
        return -(
            torch.logsumexp(positive_logits, dim=1)
            - torch.logsumexp(logits, dim=1)
        ).mean()

    def bidirectional(
        self, logits, text_group_ids, video_group_ids=None
    ):
        """Compute T2V/V2T loss while reusing one positive mask."""

        if logits.dim() != 2:
            raise ValueError(
                f"bidirectional multi-positive logits must be 2D, got "
                f"shape={tuple(logits.shape)}"
            )
        text_group_ids = self._prepare_group_ids(
            text_group_ids, "text_group_ids", logits.device
        )
        if video_group_ids is None:
            video_group_ids = text_group_ids
        else:
            video_group_ids = self._prepare_group_ids(
                video_group_ids, "video_group_ids", logits.device
            )
        expected_shape = (
            text_group_ids.numel(),
            video_group_ids.numel(),
        )
        if tuple(logits.shape) != expected_shape:
            raise ValueError(
                f"logit/group shape mismatch: logits={tuple(logits.shape)} "
                f"text={text_group_ids.numel()} "
                f"video={video_group_ids.numel()}"
            )

        positive_mask = text_group_ids[:, None].eq(
            video_group_ids[None, :]
        )
        self._validate_positive_rows(
            positive_mask, text_group_ids, "text"
        )
        self._validate_positive_rows(
            positive_mask.T, video_group_ids, "video"
        )
        text_to_video = self._loss_from_positive_mask(
            logits, positive_mask
        )
        # Keep the frozen P0 transpose-and-row-reduction path.  Reducing the
        # original tensor over dim=0 is mathematically equivalent but changes
        # floating-point gradient accumulation at the 1e-6 level in FP16.
        video_to_text = self._loss_from_positive_mask(
            logits.T, positive_mask.T
        )
        return (text_to_video + video_to_text) / 2, positive_mask

    @staticmethod
    def _validate_positive_rows(positive_mask, group_ids, direction):
        missing = (~positive_mask.any(dim=1)).nonzero(as_tuple=False).reshape(-1)
        if missing.numel():
            missing_groups = group_ids[missing]
            raise ValueError(
                f"multi-positive target missing positive {direction} "
                f"indices={missing.tolist()} "
                f"group IDs={missing_groups.tolist()}"
            )

    @staticmethod
    def _loss_from_positive_mask(logits, positive_mask):
        positive_logits = logits.masked_fill(
            ~positive_mask, float("-inf")
        )
        return -(
            torch.logsumexp(positive_logits, dim=1)
            - torch.logsumexp(logits, dim=1)
        ).mean()


class MILNCELoss(nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()
        # self.batch_size = batch_size
        # self.n_pair = n_pair
        torch_v = float(".".join(torch.__version__.split(".")[:2]))
        self.bool_dtype = torch.bool if torch_v >= 1.3 else torch.uint8

    def forward(self, sim_matrix, batch_size, n_pair):
        mm_mask = np.eye(batch_size)
        mm_mask = np.kron(mm_mask, np.ones((n_pair, n_pair)))  # 克罗克内积
        mm_mask = torch.tensor(mm_mask).float().to(sim_matrix.device)

        from_text_matrix = sim_matrix + mm_mask * -1e12
        from_video_matrix = sim_matrix.transpose(1, 0)

        new_sim_matrix = torch.cat([from_video_matrix, from_text_matrix], dim=-1)
        logpt = F.log_softmax(new_sim_matrix, dim=-1)

        mm_mask_logpt = torch.cat([mm_mask, torch.zeros_like(mm_mask)], dim=-1)
        masked_logpt = logpt + (torch.ones_like(mm_mask_logpt) - mm_mask_logpt) * -1e12

        new_logpt = -torch.logsumexp(masked_logpt, dim=-1)

        logpt_choice = torch.zeros_like(new_logpt)
        mark_ind = torch.arange(batch_size).to(sim_matrix.device) * n_pair + (
            n_pair // 2
        )
        logpt_choice[mark_ind] = 1
        sim_loss = new_logpt.masked_select(
            logpt_choice.to(dtype=self.bool_dtype)
        ).mean()
        return sim_loss


class MILNCELoss_BoF(nn.Module):
    def __init__(self):
        super(MILNCELoss_BoF, self).__init__()
        # self.batch_size = batch_size
        # self.n_pair = n_pair
        torch_v = float(".".join(torch.__version__.split(".")[:2]))
        self.bool_dtype = torch.bool if torch_v >= 1.3 else torch.uint8

    def forward(self, sim_matrix, batch_size, n_video, n_text):
        if sim_matrix.size(0) // batch_size == n_video:  # from v
            la = np.ones((n_video, n_text))
        else:
            la = np.ones((n_text, n_video))

        mm_mask = np.eye(batch_size)
        mm_mask = np.kron(mm_mask, la)  # 克罗克内积
        mm_mask = torch.tensor(mm_mask).float().bool()
        mm_mask = mm_mask.to(sim_matrix.device)

        sim_loss = -(F.log_softmax(sim_matrix, dim=1) * mm_mask).sum(1) / mm_mask.sum(1)
        sim_loss = sim_loss.mean()
        return sim_loss


class KLdivergence(nn.Module):
    def __init__(self):
        super(KLdivergence, self).__init__()

    def kl_divergence(self, mu, logsigma):
        # Treat `logsigma` as log-variance (logvar = log(sigma^2)).
        # Use a stable reduction (mean over batch) to avoid loss scale exploding with batch/D.
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=-1).mean()

    def forward(
        self,
        sampled_video_features,
        video_logsigma,
        sampled_text_features,
        text_logsigma,
    ):
        vib_loss = self.kl_divergence(
            sampled_video_features.mean(dim=1), video_logsigma
        ) + self.kl_divergence(sampled_text_features.mean(dim=1), text_logsigma)
        return vib_loss


class MaxMarginRankingLoss(nn.Module):
    def __init__(
        self,
        margin=1.0,
        negative_weighting=False,
        batch_size=1,
        n_pair=1,
        hard_negative_rate=0.5,
    ):
        super(MaxMarginRankingLoss, self).__init__()
        self.margin = margin
        self.n_pair = n_pair
        self.batch_size = batch_size
        easy_negative_rate = 1 - hard_negative_rate
        self.easy_negative_rate = easy_negative_rate
        self.negative_weighting = negative_weighting
        if n_pair > 1 and batch_size > 1:
            alpha = easy_negative_rate / ((batch_size - 1) * (1 - easy_negative_rate))
            mm_mask = (1 - alpha) * np.eye(self.batch_size) + alpha
            mm_mask = np.kron(mm_mask, np.ones((n_pair, n_pair)))
            mm_mask = torch.tensor(mm_mask) * (batch_size * (1 - easy_negative_rate))
            self.mm_mask = mm_mask.float()

    def forward(self, x):
        d = torch.diag(x)
        max_margin = F.relu(self.margin + x - d.view(-1, 1)) + F.relu(
            self.margin + x - d.view(1, -1)
        )
        if self.negative_weighting and self.n_pair > 1 and self.batch_size > 1:
            max_margin = max_margin * self.mm_mask.to(max_margin.device)
        return max_margin.mean()


def _use_contiguous_distributed_collectives():
    """Use allocation-friendly collectives only on a supported backend.

    ``all_gather_into_tensor``/``reduce_scatter_tensor`` avoid Python lists and
    an extra concatenation on the production NCCL path.  The Gloo version
    bundled with the project's PyTorch build does not implement the underlying
    base collectives, so CPU tests and diagnostics must retain the portable
    list/all-reduce implementation.
    """

    if not (
        hasattr(torch.distributed, "all_gather_into_tensor")
        and hasattr(torch.distributed, "reduce_scatter_tensor")
    ):
        return False
    try:
        backend = str(torch.distributed.get_backend()).lower()
    except (RuntimeError, ValueError):
        return False
    return backend == "nccl" or backend.endswith(".nccl")


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        world_size, rank = _distributed_context(args)
        tensor = tensor.contiguous()
        use_contiguous_collectives = _use_contiguous_distributed_collectives()
        if use_contiguous_collectives:
            output = tensor.new_empty(
                (world_size * tensor.shape[0], *tensor.shape[1:])
            )
            torch.distributed.all_gather_into_tensor(output, tensor)
        else:
            output_list = [torch.empty_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(output_list, tensor)
            output = torch.cat(output_list, dim=0)
        ctx.args = args
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        ctx.use_contiguous_collectives = use_contiguous_collectives
        return output

    @staticmethod
    def backward(ctx, grad_output):
        world_size, rank = _distributed_context(ctx.args)
        if world_size != ctx.world_size or rank != ctx.rank:
            raise RuntimeError(
                "distributed context changed between all-gather forward and backward: "
                f"forward world_size={ctx.world_size}, rank={ctx.rank}; "
                f"backward world_size={world_size}, rank={rank}"
            )
        grad_output = grad_output.contiguous()
        if ctx.use_contiguous_collectives:
            local_gradient = grad_output.new_empty(
                (ctx.batch_size, *grad_output.shape[1:])
            )
            torch.distributed.reduce_scatter_tensor(
                local_gradient,
                grad_output,
                op=torch.distributed.ReduceOp.SUM,
            )
        else:
            torch.distributed.all_reduce(
                grad_output, op=torch.distributed.ReduceOp.SUM
            )
            offset = rank * ctx.batch_size
            local_gradient = grad_output.narrow(
                0, offset, ctx.batch_size
            ).contiguous()
        return (
            local_gradient,
            None,
        )


def _distributed_context(args):
    configured_world_size = getattr(args, "world_size", None)
    configured_rank = getattr(args, "rank", None)
    initialized = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )

    if initialized:
        actual_world_size = torch.distributed.get_world_size()
        actual_rank = torch.distributed.get_rank()
        world_size = (
            actual_world_size
            if configured_world_size is None
            else int(configured_world_size)
        )
        rank = actual_rank if configured_rank is None else int(configured_rank)
        if world_size != actual_world_size or rank != actual_rank:
            raise RuntimeError(
                "distributed task config does not match initialized process group: "
                f"configured world_size={world_size}, rank={rank}; "
                f"actual world_size={actual_world_size}, rank={actual_rank}"
            )
    else:
        world_size = (
            1 if configured_world_size is None else int(configured_world_size)
        )
        rank = 0 if configured_rank is None else int(configured_rank)

    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"rank must be in [0, {world_size}), got rank={rank}"
        )
    if world_size > 1 and not initialized:
        raise RuntimeError(
            f"distributed all-gather requested with world_size={world_size}, "
            "but the process group is not initialized"
        )
    return world_size, rank


def allgather_no_grad(tensor, args):
    world_size, _rank = _distributed_context(args)
    if world_size == 1:
        return tensor.detach()
    with torch.no_grad():
        tensor = tensor.contiguous()
        if _use_contiguous_distributed_collectives():
            output = tensor.new_empty(
                (world_size * tensor.shape[0], *tensor.shape[1:])
            )
            torch.distributed.all_gather_into_tensor(output, tensor)
            return output
        output_list = [torch.empty_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(output_list, tensor)
        return torch.cat(output_list, dim=0)


def allgather_with_grad(tensor, args):
    world_size, _rank = _distributed_context(args)
    if world_size == 1:
        return tensor
    return AllGather.apply(tensor, args)


class dual_softmax_loss(nn.Module):
    def __init__(
        self,
    ):
        super(dual_softmax_loss, self).__init__()

    def forward(self, sim_matrix, temp=1000):
        sim_matrix = sim_matrix * F.softmax(sim_matrix / temp, dim=0) * len(sim_matrix)
        # With an appropriate temperature parameter, the model achieves higher performance
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        loss = -logpt
        return loss


if __name__ == "__main__":
    sim_matrix = torch.randn(12, 12)
    loss_fct = MILNCELoss_BoF()
    loss = loss_fct(sim_matrix, batch_size=4, n_pair=3)
