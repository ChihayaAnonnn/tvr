"""Explicit fusion core; legacy and vendor algorithms remain untouched."""
from dataclasses import dataclass
import torch
import torch.distributed as dist
from .core import AqsCore, _ContiguousBackward
from .selection import pooled_frames
from .fusion_math import score_gaussian, symmetric_kl, fuse_scores
from semantic_retrieval.full_data.upstream_uatvr.modeling import allgather
from semantic_retrieval.full_data.upstream_uatvr.probability.tensor_utils import l2_normalize, sample_gaussian_tensors
from semantic_retrieval.full_data.worker import _collective_call


@dataclass(frozen=True)
class PreparedModal:
    pooled: torch.Tensor
    tokens: torch.Tensor
    mask: torch.Tensor
    real_count: int


@dataclass(frozen=True)
class FusionComponents:
    wti: torch.Tensor
    distance: torch.Tensor
    raw: torch.Tensor
    logits: torch.Tensor
    mil: torch.Tensor
    kl: torch.Tensor
    representation: torch.Tensor
    stats: dict


class FusionCore(AqsCore):
    def _local(self,call):
        return _collective_call(call) if self.training else call()

    def _adapt_local(self,text,video):
        if self.sim_header != 'seqTransf': raise ValueError('fusion requires seqTransf')
        text_token=text['tokens'].contiguous()
        visual_output=video['tokens'].contiguous()
        attention_mask=text['mask'].long(); video_mask=video['mask'].long()
        for tokens,mask in ((text_token,attention_mask),(visual_output,video_mask)):
            if (tokens.ndim!=3 or mask.shape!=tokens.shape[:2] or not ((mask==0)|(mask==1)).all()
                    or (mask.sum(1)==0).any() or not torch.isfinite(tokens).all()):
                raise ValueError('invalid fusion token/mask contract')
        visual_output_original = visual_output

        extra_token_num = self.extra_cls_frame_num
        seq_length = visual_output.size(1) + extra_token_num     # extra 2 learnable token
        position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
        position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)    # bs num+extra_token_num dim
        frame_position_embeddings[:, 0:visual_output.size(1), :] += visual_output
        visual_output = frame_position_embeddings

        tempo_mask = torch.cat([video_mask, torch.ones(visual_output.size(0), extra_token_num).to(visual_output.device)], axis=1)
        extended_video_mask = (1.0 - tempo_mask.unsqueeze(1)) * -1000000.0
        extended_video_mask = extended_video_mask.expand(-1, tempo_mask.size(1), -1)
        visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
        visual_output = self.transformerClip(visual_output, extended_video_mask)
        visual_output = visual_output.permute(1, 0, 2).contiguous()  # LND -> NLD
        # multi extra token
        # visual_output = visual_output[:, :visual_output_original.size(1), :] + visual_output_original   # residual fusion
        visual_output[:, :visual_output_original.size(1), :] += visual_output_original
        video_mask = tempo_mask

        # sequential type: MLP for text with extra token
        text_original = text_token  # save original
        extra_text_num = self.extra_cls_text_num
        seq_text_length = extra_text_num + text_token.size(1)
        position_ids_text = torch.arange(seq_text_length, dtype=torch.long, device=text_token.device)
        position_ids_text = position_ids_text.unsqueeze(0).expand(text_token.size(0), -1)
        word_position_embeddings = self.word_position_embeddings(position_ids_text)
        word_position_embeddings[:, 0:text_token.size(1), :] += text_token
        text_token = word_position_embeddings

        tempo_mask_ = torch.cat([attention_mask, torch.ones(text_token.size(0), extra_text_num).to(text_token.device)], axis=1)
        extended_text_mask = (1.0 - tempo_mask_.unsqueeze(1)) * -1000000.0
        extended_text_mask = extended_text_mask.expand(-1, tempo_mask_.size(1), -1)
        text_token = text_token.permute(1, 0, 2)
        text_token = self.transformerClip(text_token, extended_text_mask)
        text_token = text_token.permute(1, 0, 2).contiguous()
        text_token[:, :text_original.size(1), :] += text_original
        attention_mask = tempo_mask_


        return text_token,attention_mask,visual_output,video_mask

    def prepare_modalities(self,text,video,*,gather):
        if type(gather) is not bool or (gather and not self.training):
            raise ValueError('invalid fusion gather mode')
        values=self._local(lambda:self._adapt_local(text,video))
        if gather and dist.is_initialized():
            shapes=[None]*dist.get_world_size()
            dist.all_gather_object(shapes,[(tuple(x.shape),str(x.dtype)) for x in values])
            if any(s!=shapes[0] for s in shapes): raise ValueError('fusion rank tensor shapes differ')
            values=tuple(allgather(x,self.task_config) for x in values)
        tt,tm,vt,vm=values
        tt=_ContiguousBackward.apply(tt) if gather else tt
        vt=_ContiguousBackward.apply(vt) if gather else vt
        def prepare():
            tt_norm=tt/tt.norm(dim=-1,keepdim=True)
            vt_norm=vt/vt.norm(dim=-1,keepdim=True)
            nt,nv=text['tokens'].shape[1],video['tokens'].shape[1]
            # A singleton slice can alias the full mask in the upstream helper.
            tp=self._mean_pooling_for_similarity_sequence(tt_norm[:,:nt].contiguous(),tm[:,:nt].clone())
            vp=self._mean_pooling_for_similarity_visual(vt_norm[:,:nv].contiguous(),vm[:,:nv].clone())
            tp=tp/tp.norm(dim=-1,keepdim=True); vp=vp/vp.norm(dim=-1,keepdim=True)
            if any(not torch.isfinite(x).all() for x in (tt_norm,vt_norm,tp,vp)):
                raise ValueError('nonfinite prepared fusion features')
            return PreparedModal(tp,tt_norm,tm,nt),PreparedModal(vp,vt_norm,vm,nv)
        return self._local(prepare)

    def distribution_parameters(self,modal,*,modality):
        if modality not in ('text','video'): raise ValueError('invalid fusion modality')
        features=modal.tokens[:,:modal.real_count].contiguous()
        mean,_,_=getattr(self,'pie_net_'+modality)(modal.pooled,features)
        a=getattr(self,'uncertain_net_'+modality)(modal.pooled,features)['logsigma']
        return dict(mean=l2_normalize(mean),logsigma=a)

    def score_prepared(self,text,video):
        tp=self.distribution_parameters(text,modality='text')
        vp=self.distribution_parameters(video,modality='video')
        distance=symmetric_kl(score_gaussian(**tp),score_gaussian(**vp))
        wti=self.weighted_token_wise_intersection(text.tokens,video.tokens,text.mask,video.mask)
        raw,logits=fuse_scores(wti,distance,self.clip.logit_scale,self.fusion_config)
        return dict(wti=wti,distance=distance,raw=raw,logits=logits,text_params=tp,video_params=vp)

    def fusion_components(self,text,video):
        if not self.training or not self.fusion_config.active:
            raise ValueError('fusion components require active training')
        t,v=self.prepare_modalities(text,video,gather=True)
        def calculate():
            scores=self.score_prepared(t,v)
            tp,vp=scores['text_params'],scores['video_params']
            ve=sample_gaussian_tensors(vp['mean'],vp['logsigma'],self.n_video_samples)
            te=sample_gaussian_tensors(tp['mean'],tp['logsigma'],self.n_text_samples)
            b,d=ve.shape[0],ve.shape[-1]
            similarity=ve.reshape(-1,d)@te.reshape(-1,d).T
            mil=(self.loss_MIL_fct(similarity,b,self.n_video_samples,self.n_text_samples)
                 +self.loss_MIL_fct(similarity.T,b,self.n_text_samples,self.n_video_samples))/2
            kl=self.vib_loss(ve,vp['logsigma'],te,tp['logsigma'])
            stats={}
            for name,params in (('text',tp),('video',vp)):
                a=params['logsigma'].detach().float()
                stats.update({name+'_logvar_low_fraction':float((a<=-4).float().mean()),
                              name+'_logvar_high_fraction':float((a>=2).float().mean())})
                for q in (.1,.5,.9): stats[f'{name}_logvar_q{int(q*100)}']=float(torch.quantile(2*a.clamp(-4,2),q))
            for name,value in (('wti',scores['wti']),('distance',scores['distance']),
                               ('eta_distance',self.fusion_config.eta*scores['distance'])):
                detached=value.detach()
                stats.update({name+'_mean':float(detached.mean()),name+'_min':float(detached.min()),
                              name+'_max':float(detached.max()),name+'_std':float(detached.std(unbiased=False))})
            return FusionComponents(scores['wti'],scores['distance'],scores['raw'],scores['logits'],
                .01*mil,.0001*kl,pooled_frames(v.tokens,v.mask,v.real_count),stats)
        return self._local(calculate)

    def _loose_similarity(self,sequence_output,text_token,visual_output,attention_mask,video_mask,
                          sim_header='seqTransf',*,training_observer=None):
        if not self.fusion_config.active:
            return super()._loose_similarity(sequence_output,text_token,visual_output,attention_mask,
                video_mask,sim_header=sim_header,training_observer=training_observer)
        if sim_header!='seqTransf': raise ValueError('fusion requires seqTransf')
        text=dict(pooled=sequence_output,tokens=text_token,mask=attention_mask)
        video=dict(tokens=visual_output,mask=video_mask)
        if self.training:
            result=self.fusion_components(text,video)
            if training_observer is not None:
                # The old observer contract is unavailable for active fusion.
                raise ValueError('use fusion_components for active fusion')
            return result.logits,result.mil,result.kl
        if training_observer is not None: raise ValueError('observer is training-only')
        t,v=self.prepare_modalities(text,video,gather=False)
        return self.score_prepared(t,v)['logits']
