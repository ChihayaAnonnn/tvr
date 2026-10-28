"""Trainable UATVR adapter with a training-only video-level AQS objective."""
import re
import torch
import torch.distributed as dist
import torch.nn.functional as F

from semantic_retrieval.full_data.models import BaselineModel
from semantic_retrieval.full_data.worker import _collective_call
from semantic_retrieval.runtime.distributed import gather_metadata
from .config import AqsConfig
from .core import AqsCore
from .loss import selected_t2v_loss
from .selection import select_negatives, summarize_selection, validate_ids


class AqsBaselineModel(BaselineModel):
    def __init__(self, clip_config, clip_state, *, aqs, id_mapping_sha256, **baseline_kwargs):
        def validate():
            if (not isinstance(aqs, AqsConfig) or not isinstance(id_mapping_sha256, str)
                    or re.fullmatch('[0-9a-f]{64}', id_mapping_sha256) is None
                    or baseline_kwargs.get('batching', 'unique_video_all_captions_v1') != 'unique_video_all_captions_v1'):
                raise ValueError('invalid AQS model configuration or ID mapping')
        # No collectives exist inside initialization; local setup errors are
        # synchronized before entering the rank-context collective below.
        if dist.is_initialized():
            _collective_call(validate)
        else:
            validate()
        _collective_call(lambda: super(AqsBaselineModel, self).__init__(
            'uatvr', clip_config, clip_state, **baseline_kwargs))
        self.aqs_config = aqs
        self.id_mapping_sha256 = id_mapping_sha256
        self.validate_rank_context()

    @staticmethod
    def _uatvr_core_class():
        return AqsCore

    def validate_rank_context(self):
        """Call on every rank after creating a process group, before wrapping DDP."""
        context = (self.aqs_config.recipe(), self.id_mapping_sha256)
        if dist.is_initialized():
            records = [None]*dist.get_world_size()
            dist.all_gather_object(records, context)
            if any(record != context for record in records):
                raise ValueError('AQS rank configuration or ID mapping mismatch')
            self._aqs_group = dist.group.WORLD
        else:
            self._aqs_group = None
        self._aqs_context = context

    def _validate_batch(self, input_ids, text_mask, video, frame_mask, video_ids):
        tensors = (input_ids, text_mask, video, frame_mask, video_ids)
        if any(not isinstance(value, torch.Tensor) or value.ndim < 1 for value in tensors):
            raise ValueError('invalid paired AQS tensors')
        if (not self.training or len(input_ids) < 1 or video_ids.dtype != torch.long
                or video_ids.shape != (len(input_ids),) or len(video) != len(input_ids)
                or any(value.device != input_ids.device for value in tensors)
                or (video_ids < 0).any()):
            raise ValueError('paired AQS training input contract mismatch')
        if self._aqs_context != (self.aqs_config.recipe(), self.id_mapping_sha256):
            raise ValueError('AQS configuration changed after rank validation')
        group = dist.group.WORLD if dist.is_initialized() else None
        if group is not self._aqs_group:
            raise ValueError('call validate_rank_context after process-group initialization')

    def forward(self, input_ids, text_mask, video, frame_mask, video_ids):
        if not self.training or not self.aqs_config.active:
            return super().forward(input_ids, text_mask, video, frame_mask, video_ids)
        _collective_call(lambda: self._validate_batch(input_ids, text_mask, video, frame_mask, video_ids))
        global_ids = gather_metadata(video_ids)
        validate_ids(global_ids, len(global_ids))
        text, visual = _collective_call(lambda: (
            self.encode_text(input_ids, text_mask), self.encode_video(video, frame_mask)))
        # Existing UATVR collectives produce globally aligned WTI inputs once.
        logits, mil, kl, raw, representation = self.core.training_components(text, visual)
        if logits.shape != (len(global_ids), len(global_ids)):
            raise ValueError('global AQS paired loss shape mismatch')
        target = torch.arange(len(global_ids), device=logits.device)
        contrastive = (F.cross_entropy(logits, target)+F.cross_entropy(logits.T, target))/2
        selection = select_negatives(raw, representation, global_ids, self.aqs_config.negatives)
        aqs = selected_t2v_loss(logits, selection)
        loss = contrastive+mil+kl+self.aqs_config.weight*aqs
        if not torch.isfinite(loss):
            raise ValueError('nonfinite AQS training loss')
        return dict(loss=loss, contrastive=contrastive, mil=mil, kl=kl, logits=logits,
                    aqs=aqs, aqs_stats=summarize_selection(selection))
