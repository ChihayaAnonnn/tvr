"""Real Laya adapter; image features are reusable, query states never are."""
from pathlib import Path
import numpy as np
import torch

QUESTION = 'Based only on the sampled frames, does this video match the following retrieval description: {}?'
CRITERIA = {'match': 'The sampled visual evidence supports the description.',
            'not_supported': 'The sampled visual evidence does not establish the description.'}
STATE = {'context': 'These are sampled frames from the same video, ordered chronologically.'}


class LayaJudge:
    def __init__(self, weights, device='cpu'):
        from laya.vlm import VLMAgent
        root = Path(weights).resolve(strict=True)
        for item in ('model.safetensors', 'vlm_agent_config.json', 'backbone/config.json', 'processor'):
            if not (root/item).exists():
                raise ValueError('complete local checkpoint required: '+item)
        self.agent = VLMAgent(str(root), device=device, dtype='fp32')
        self.agent.model.eval().requires_grad_(False)
        if self.agent.cfg['option_attention'] != 'block':
            raise ValueError('unexpected checkpoint attention policy')

    @torch.inference_mode()
    def encode(self, images):
        from laya.vlm import vlm_prefix
        if not 1 <= len(images) <= 4:
            raise ValueError('one to four real frames required')
        a = self.agent
        prefix = vlm_prefix(a.processor, images, a.prep)
        features = a.model.encode_images(prefix['pixel_values'].to(a.device, torch.float32),
                                        prefix['pixel_attention_mask'].to(a.device))
        return {'ids': list(prefix['ids']), 'features': features.cpu().contiguous()}

    @torch.inference_mode()
    def score(self, queries, evidence):
        from laya.vlm import build_vlm_inputs, collate_vlm, QTYPES
        from laya.common import truncation_answer
        if not queries or len(queries) != len(evidence) or len(queries) > 8:
            raise ValueError('one to eight paired queries and evidence required')
        a = self.agent
        rows = []
        for query, cache in zip(queries, evidence):
            if not isinstance(query, str) or not query.strip():
                raise ValueError('empty retrieval query')
            q = a._to_internal({'type': 'choice', 'instructions': QUESTION.format(query), 'criteria': CRITERIA})
            prefix = {'ids': cache['ids'], 'pixel_values': None, 'pixel_attention_mask': None,
                      'raw_images': None, 'n_images': len(cache['features'])}
            row = build_vlm_inputs(a.processor, STATE, q, a.cfg['max_len'], a.cfg['head_max_len'],
                                   option_order=[0, 1], prefix=prefix)
            if truncation_answer(row['truncation'], q):
                raise ValueError('truncated retrieval evidence or query')
            row['qtype'] = QTYPES['choice']; rows.append(row)
        batch = collate_vlm(rows, a.processor.tokenizer.pad_token_id, with_pixels=False)
        args = [batch[k].to(a.device) for k in ('input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype')]
        features = torch.cat([c['features'] for c in evidence]).to(a.device, torch.float32)
        logits, _ = a.model(*args, image_hidden_states=features, option_span=batch['option_span'].to(a.device))
        z = logits.float().cpu().numpy()
        if z.shape != (len(queries), 2) or not np.isfinite(z).all():
            raise ValueError('invalid Laya logits')
        t = a._checkpoint_temperature(QTYPES['choice'], 2)
        if not np.isfinite(t) or t <= 0: raise ValueError('invalid checkpoint temperature')
        scaled = z.astype(np.float64)/t
        p = np.exp(scaled-scaled.max(axis=1, keepdims=True)); p /= p.sum(axis=1, keepdims=True)
        return z, p[:, 0]
