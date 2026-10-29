"""Independent spatial-query adapter preserving the original AQS core."""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from semantic_retrieval.types import EncoderInputs
from semantic_retrieval.full_data.worker import _collective_call
from semantic_retrieval.runtime.distributed import gather_with_grad, gather_metadata
from .model import AqsBaselineModel
from .fusion_config import FusionConfig
from .local_config import LocalQueryConfig
from .local_tokens import LocalQueryTokens, lexical_mask
from .local_visual import visual_tokens
from .local_scoring import local_scores, add_local_logits
from .selection import validate_ids, select_negatives, summarize_selection
from .loss import selected_t2v_loss


def _settings(local, fusion, query_chunk, video_chunk):
    if not isinstance(local,LocalQueryConfig) or not isinstance(fusion,FusionConfig):
        raise ValueError('invalid local/fusion configuration')
    if fusion.enabled or fusion.eta != 0:
        raise ValueError('local query v1 requires new KL fusion disabled and eta=0')
    if any(type(n) is not int or n < 1 for n in (query_chunk,video_chunk)):
        raise ValueError('invalid local scoring chunk sizes')


def build_local_model(clip_config, clip_state, *, local, fusion, aqs, id_mapping_sha256,
                      query_chunk=8, video_chunk=8, **baseline_kwargs):
    _collective_call(lambda: _settings(local,fusion,query_chunk,video_chunk))
    if not local.active:
        return AqsBaselineModel(clip_config,clip_state,aqs=aqs,id_mapping_sha256=id_mapping_sha256,**baseline_kwargs)
    return LocalQueryAqsModel(clip_config,clip_state,local=local,fusion=fusion,aqs=aqs,
        id_mapping_sha256=id_mapping_sha256,query_chunk=query_chunk,video_chunk=video_chunk,**baseline_kwargs)


class LocalQueryAqsModel(AqsBaselineModel):
    def __init__(self,clip_config,clip_state,*,local,fusion,query_chunk=8,video_chunk=8,**kwargs):
        def validate():
            _settings(local,fusion,query_chunk,video_chunk)
            if not local.active: raise ValueError('use factory for inactive local query')
            if (clip_config.image_size//clip_config.patch_size != 7
                    or clip_config.embed_dim % 8):
                raise ValueError('local model requires 7x7 patches and eight attention heads')
        _collective_call(validate)
        self.local_config, self.fusion_config = local,fusion
        self.query_chunk,self.video_chunk = query_chunk,video_chunk
        super().__init__(clip_config,clip_state,**kwargs)
        self.local_queries = LocalQueryTokens(clip_config.embed_dim)

    def _context(self):
        return (self.local_config.recipe(),self.fusion_config.recipe(),self.query_chunk,self.video_chunk)

    def validate_rank_context(self):
        super().validate_rank_context()
        context=self._context()
        if dist.is_initialized():
            records=[None]*dist.get_world_size()
            dist.all_gather_object(records,context)
            if any(record != context for record in records): raise ValueError('local rank configuration mismatch')
        self._local_context=context

    def _validate_local_context(self):
        _settings(self.local_config,self.fusion_config,self.query_chunk,self.video_chunk)
        if (self._context() != self._local_context
                or self._aqs_context != (self.aqs_config.recipe(),self.id_mapping_sha256)):
            raise ValueError('local or AQS configuration changed after rank validation')
        group=dist.group.WORLD if dist.is_initialized() else None
        if group is not self._aqs_group: raise ValueError('call validate_rank_context after group initialization')

    def encode_text(self,input_ids,mask):
        result=super().encode_text(input_ids,mask)
        return dict(result,local_tokens=result['tokens'],local_mask=lexical_mask(mask))

    def encode_video(self,video,mask):
        cfg=self.clip_config
        EncoderInputs(torch.tensor([[cfg.vocab_size-2,cfg.vocab_size-1]],device=video.device),
            torch.ones((1,2),dtype=torch.bool,device=video.device),video,mask)
        if video.shape[-2:] != (cfg.image_size,cfg.image_size) or video.shape[1]>12:
            raise ValueError('invalid local video spatial/frame dimensions')
        b,f=video.shape[:2]
        cls,patches=visual_tokens(self.clip.visual,video.flatten(0,1),video_frame=f)
        local=self.local_queries(patches.reshape(b,f,49,-1),mask)
        return dict(tokens=cls.reshape(b,f,-1),mask=mask,local_tokens=local)

    def _local_raw(self,text,video):
        return local_scores(text['local_tokens'],text['local_mask'],video['local_tokens'],video['mask'],
            tau=self.local_config.tau,query_chunk=self.query_chunk,video_chunk=self.video_chunk)

    def _logits(self,text,video):
        if self.training: raise ValueError('local training uses explicit forward components')
        self._validate_local_context()
        global_logits=super()._logits(text,video)[0]
        return add_local_logits(global_logits,self._local_raw(text,video),self.clip.logit_scale,
                                beta=self.local_config.beta),None,None

    def _validate_encoded(self,text,video):
        for tensor in (text['tokens'],text['pooled'],video['tokens'],video['local_tokens']):
            if not torch.isfinite(tensor).all(): raise ValueError('nonfinite encoded local inputs')

    def _rank_shapes(self,text,video):
        tensors=(text['tokens'],text['mask'],video['tokens'],video['mask'],video['local_tokens'])
        specs=tuple((tuple(t.shape),str(t.dtype),t.device.type) for t in tensors)
        if dist.is_initialized():
            records=[None]*dist.get_world_size();dist.all_gather_object(records,specs)
            if any(r != specs for r in records): raise ValueError('local rank feature shape mismatch')

    def forward(self,input_ids,text_mask,video,frame_mask,video_ids):
        def validate():
            self._validate_local_context()
            self._validate_batch(input_ids,text_mask,video,frame_mask,video_ids)
        _collective_call(validate)
        ids=gather_metadata(video_ids);validate_ids(ids,len(ids))
        text,visual=_collective_call(lambda:(self.encode_text(input_ids,text_mask),self.encode_video(video,frame_mask)))
        _collective_call(lambda:self._validate_encoded(text,visual))
        self._rank_shapes(text,visual)
        global_logits,mil,kl,global_raw,representation=_collective_call(lambda:self.core.training_components(text,visual))
        lt=gather_with_grad(text['local_tokens']); words=gather_metadata(text['local_mask'])
        lv=gather_with_grad(visual['local_tokens']); frames=gather_metadata(visual['mask'])
        local_raw=_collective_call(lambda:local_scores(lt,words,lv,frames,tau=self.local_config.tau,
            query_chunk=self.query_chunk,video_chunk=self.video_chunk))
        logits=_collective_call(lambda:add_local_logits(global_logits,local_raw,self.clip.logit_scale,beta=self.local_config.beta))
        def losses():
            if logits.shape != (len(ids),len(ids)): raise ValueError('local paired loss shape mismatch')
            target=torch.arange(len(ids),device=logits.device)
            ce=(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))/2
            aqs=logits.sum()*0
            stats=dict(mean_negatives=0.,mean_difficulty=0.,mean_uniqueness=0.,empty_rows=0.)
            if self.aqs_config.active:
                selection=select_negatives(global_raw+self.local_config.beta*local_raw.detach(),representation,ids,self.aqs_config.negatives)
                aqs=selected_t2v_loss(logits,selection);stats=summarize_selection(selection)
            loss=ce+mil+kl+self.aqs_config.weight*aqs
            if not torch.isfinite(loss): raise ValueError('nonfinite local training loss')
            return dict(loss=loss,contrastive=ce,mil=mil,kl=kl,logits=logits,aqs=aqs,aqs_stats=stats,
                local_stats=dict(score_mean=float(local_raw.detach().mean()),
                    lexical_tokens=float(words.sum()),valid_slots=float(frames.sum()*9)))
        return _collective_call(losses)
