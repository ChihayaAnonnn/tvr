"""Explicit UATVR training extension; vendor sources remain unchanged.

The method below is copied from full_data/upstream_uatvr/modeling.py at
976f815, with only a keyword-only observer and its call added. An AST
regression compares it to the original after removing these additions.
"""
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from semantic_retrieval.full_data.upstream_uatvr.modeling import UATVR, allgather
from .selection import pooled_frames


class AqsCore(UATVR):
    def _loose_similarity(self, sequence_output, text_token, visual_output, attention_mask, video_mask, sim_header="seqTransf", *, training_observer=None):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        frame_num = visual_output.size(1)   # 12 / 64
        word_num = text_token.size(1)       # 32 /

        if sim_header == "meanP":
            # Default: Parameter-free type
            pass
        elif sim_header == "seqLSTM":
            # Sequential type: LSTM
            visual_output_original = visual_output
            visual_output = pack_padded_sequence(visual_output, torch.sum(video_mask, dim=-1).cpu(),
                                                 batch_first=True, enforce_sorted=False)
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training: self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
            visual_output = torch.cat((visual_output, visual_output_original[:, visual_output.size(1):, ...].contiguous()), dim=1)
            visual_output = visual_output + visual_output_original
        elif sim_header == "seqTransf":
            # Sequential type: Transformer Encoder +++++++++++++= extra token
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

        if self.training:       # works only on ddp
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)      # Bxgpu 1 dim
            text_token = allgather(text_token, self.task_config)        # Bxgpu 32 dim
            attention_mask = allgather(attention_mask, self.task_config)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_pooled = self._mean_pooling_for_similarity_visual(visual_output[:, 0:frame_num, :].contiguous(), video_mask[:, 0:frame_num].contiguous())
        visual_pooled = visual_pooled / visual_pooled.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        text_token = text_token / text_token.norm(dim=-1, keepdim=True)
        text_pooled = self._mean_pooling_for_similarity_sequence(text_token[:, 0:word_num, :].contiguous(), attention_mask[:, 0:word_num].contiguous())
        text_pooled = text_pooled / text_pooled.norm(dim=-1, keepdim=True)

        # ti_logits = self.token_wise_interaction(text_token=text_token, frame_token=visual_output, attention_mask=attention_mask, video_mask=video_mask)
        wti_logits = self.weighted_token_wise_intersection(text_token, visual_output, attention_mask, video_mask)
        if training_observer is not None:
            if not self.training:
                raise ValueError('AQS observer is training-only')
            training_observer(wti_logits, visual_output, video_mask, frame_num)

        ####################################################################
        ############### probabilistic embedding modeling part ##############
        ####################################################################
        prob_video = self.probabilistic_video(visual_pooled, visual_output[:, 0:frame_num, :].contiguous())
        prob_video_embedding = prob_video['embedding']      # 从分布中采样m个embedding
        prob_video_logsigma = prob_video['logsigma']         # 方差
        prob_video_mean = prob_video['mean']

        prob_text = self.probabilistic_text(text_pooled, text_token[:, 0:word_num, :].contiguous())
        prob_text_embedding = prob_text['embedding']       # b n 512
        prob_text_logsigma = prob_text['logsigma']   # bs 512
        prob_text_mean = prob_text['mean']       # bs 512

        if self.training:
            ####################################################################
            ############### MILNCELoss_(Supervised Contrastice Loss) ##############
            ####################################################################
            bs = prob_video_embedding.size(0)
            n_video = self.n_video_samples
            n_text = self.n_text_samples
            dim = prob_video_embedding.size(-1)

            prob_sim_matrix_from_v = torch.einsum('ad,bd->ab', [prob_video_embedding.view(-1, dim), prob_text_embedding.view(-1, dim)])
            MIL_loss_v = self.loss_MIL_fct(prob_sim_matrix_from_v, bs, n_video, n_text)

            prob_sim_matrix_from_t = torch.einsum('ad,bd->ab', [prob_text_embedding.view(-1, dim), prob_video_embedding.view(-1, dim)])     # 与.t()等价
            MIL_loss_t = self.loss_MIL_fct(prob_sim_matrix_from_t, bs, n_video, n_text)
            MIL_loss = (MIL_loss_v + MIL_loss_t) / 2

            ####################################################################
            ############### MCofSoftCOntrastiveLoss ##############
            ####################################################################
            # mcsoft_loss, vib_loss = self.loss_mc_con(prob_video_embedding, prob_video_logsigma, prob_text_embedding, prob_text_logsigma)

            ####################################################################
            ############### KL Divergence Loss ##############
            ####################################################################
            vib_loss = self.vib_loss(prob_video_embedding, prob_video_logsigma, prob_text_embedding, prob_text_logsigma)

        # loss
        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * wti_logits      # sim_matrix
        if self.training:
            return retrieve_logits, 1e-2 * MIL_loss, 1e-4 * vib_loss
        else:   # for inference
            return retrieve_logits

    def training_components(self, text, video):
        if not self.training or self.sim_header != 'seqTransf':
            raise ValueError('AQS components require UATVR seqTransf training')
        observed = []

        def observe(raw, frames, mask, count):
            observed.append((raw.detach().float(), pooled_frames(frames, mask, count)))

        logits, mil, kl = self._loose_similarity(
            text['pooled'], text['tokens'], video['tokens'],
            text['mask'].long(), video['mask'].long(),
            sim_header=self.sim_header, training_observer=observe)
        if len(observed) != 1:
            raise ValueError('unexpected WTI observation count')
        return logits, mil, kl, *observed[0]
