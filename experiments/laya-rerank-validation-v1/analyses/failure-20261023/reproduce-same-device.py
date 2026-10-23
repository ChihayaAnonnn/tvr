import sys,os,json
from pathlib import Path
sys.dont_write_bytecode=True;sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'src'),'/data2/hxj/open-models/laya-vision/code']
import torch,numpy as np
from semantic_retrieval.laya_rerank.science import cohort
from semantic_retrieval.laya_rerank.judge import LayaJudge,STATE,QUESTION,CRITERIA
from semantic_retrieval.laya_rerank.evidence import decode_evidence
from laya.vlm import build_vlm_inputs,collate_vlm,QTYPES
r=Path.cwd();f=json.loads((r/'experiments/laya-rerank-validation-v1/fixed.json').read_text());d=cohort(r,f);torch.set_num_threads(2);torch.use_deterministic_algorithms(True);j=LayaJudge(f['weights'],'cpu');q=d['queries'][0]['query_text'];out=[]
for vi in [0,392]:
 row=d['rows'][vi];images,indices=decode_evidence(row);ev=j.encode(images);zc,pc=j.score([q],[ev]);raw={};pred=j.agent.predict({**STATE,'images':images},{'q':dict(type='choice',instructions=QUESTION.format(q),criteria=CRITERIA)},prefix_cache=False,strict=True,_raw_logits=raw)
 qi=j.agent._to_internal(dict(type='choice',instructions=QUESTION.format(q),criteria=CRITERIA));it=build_vlm_inputs(j.agent.processor,{**STATE,'images':images},qi,j.agent.cfg['max_len'],j.agent.cfg['head_max_len'],option_order=[0,1]);it['qtype']=QTYPES['choice'];b=collate_vlm([it],j.agent.processor.tokenizer.pad_token_id,with_pixels=True)
 with torch.inference_mode():zp,_=j.agent.model(*[b[k] for k in ['input_ids','attention_mask','marker_pos','marker_mask','qtype']],pixel_values=b['pixel_values'],pixel_attention_mask=b['pixel_attention_mask'],option_span=b['option_span'])
 out.append(dict(video_id=row['video_id'],native_vs_cpu_cached_max=float(abs(raw['q']-zc[0]).max()),raw_pixels_vs_cpu_cached_max=float(abs(zp.numpy()-zc).max()),probability=float(pc[0]),option_labels=list(CRITERIA),tokenizer_pad_id=j.agent.processor.tokenizer.pad_token_id,text_embedding_vocab=j.agent.model.encoder.get_input_embeddings().weight.shape[0],actual_max_input_id=max(it['ids'])))
 print(out[-1],flush=True)
Path('/tmp/laya-same-device-check.json').write_text(json.dumps(out,indent=2))
