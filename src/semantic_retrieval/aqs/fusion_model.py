"""Training and inference adapter with explicit, independently switchable fusion."""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from semantic_retrieval.full_data.worker import _collective_call
from semantic_retrieval.runtime.distributed import gather_metadata
from .model import AqsBaselineModel
from .fusion_config import FusionConfig
from .fusion_core import FusionCore
from .selection import select_negatives, summarize_selection, validate_ids
from .loss import selected_t2v_loss


class FusionAqsModel(AqsBaselineModel):
    def __init__(self,*args,fusion,**kwargs):
        def validate():
            if not isinstance(fusion,FusionConfig): raise ValueError('invalid fusion configuration')
        _collective_call(validate)
        self.fusion_config=fusion
        super().__init__(*args,**kwargs)
        self.core.fusion_config=fusion

    @staticmethod
    def _uatvr_core_class():
        return FusionCore

    def validate_rank_context(self):
        super().validate_rank_context()
        context=self.fusion_config.recipe()
        if dist.is_initialized():
            records=[None]*dist.get_world_size();dist.all_gather_object(records,context)
            if any(r!=context for r in records): raise ValueError('fusion rank configuration mismatch')
        self._fusion_context=context

    def _validate_fusion_context(self):
        if self._aqs_context != (self.aqs_config.recipe(),self.id_mapping_sha256):
            raise ValueError('AQS configuration or mapping changed after rank validation')
        if (self.fusion_config.recipe()!=self._fusion_context
                or self.core.fusion_config.recipe()!=self._fusion_context):
            raise ValueError('fusion configuration changed after rank validation')
        group=dist.group.WORLD if dist.is_initialized() else None
        if group is not self._aqs_group: raise ValueError('call validate_rank_context after process-group initialization')

    def forward(self,input_ids,text_mask,video,frame_mask,video_ids):
        _collective_call(self._validate_fusion_context)
        if not self.fusion_config.active:
            return super().forward(input_ids,text_mask,video,frame_mask,video_ids)
        _collective_call(lambda:self._validate_batch(input_ids,text_mask,video,frame_mask,video_ids))
        global_ids=gather_metadata(video_ids);validate_ids(global_ids,len(global_ids))
        text,visual=_collective_call(lambda:(self.encode_text(input_ids,text_mask),self.encode_video(video,frame_mask)))
        parts=self.core.fusion_components(text,visual)
        if parts.logits.shape!=(len(global_ids),len(global_ids)):
            raise ValueError('global fusion paired loss shape mismatch')
        target=torch.arange(len(global_ids),device=parts.logits.device)
        contrastive=(F.cross_entropy(parts.logits,target)+F.cross_entropy(parts.logits.T,target))/2
        stats=dict(mean_negatives=0.,mean_difficulty=0.,mean_uniqueness=0.,empty_rows=0.)
        auxiliary=parts.logits.sum()*0
        fusion_stats=dict(parts.stats)
        with torch.no_grad():
            d=parts.distance.detach(); b=len(d)
            fusion_stats['distance_positive_mean']=float(d.diag().mean())
            fusion_stats['distance_negative_mean']=float(d[~torch.eye(b,device=d.device,dtype=torch.bool)].mean()) if b>1 else 0.
            fusion_stats['ranking_changed_fraction']=float((parts.wti.argmax(1)!=parts.raw.argmax(1)).float().mean())
        if self.aqs_config.active:
            selection=select_negatives(parts.raw.detach(),parts.representation,global_ids,self.aqs_config.negatives)
            auxiliary=selected_t2v_loss(parts.logits,selection);stats=summarize_selection(selection)
            old=select_negatives(parts.wti.detach(),parts.representation,global_ids,self.aqs_config.negatives)
            fusion_stats['selection_changed_fraction']=float((old.selected_ids!=selection.selected_ids).float().mean()) if selection.indices.numel() else 0.
        loss=contrastive+parts.mil+parts.kl+self.aqs_config.weight*auxiliary
        if not torch.isfinite(loss): raise ValueError('nonfinite fusion loss')
        return dict(loss=loss,contrastive=contrastive,mil=parts.mil,kl=parts.kl,logits=parts.logits,
                    aqs=auxiliary,aqs_stats=stats,fusion_stats=fusion_stats)

    def score(self,text,video):
        self._validate_fusion_context()
        return super().score(text,video)
