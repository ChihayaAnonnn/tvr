import sys,os,json
from pathlib import Path
sys.dont_write_bytecode=True
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'src'),'/data2/hxj/open-models/laya-vision/code']
import numpy as np,torch
from semantic_retrieval.laya_rerank.science import cohort,read_visual,load_array
from semantic_retrieval.laya_rerank.judge import LayaJudge,QUESTION,CRITERIA,STATE
from semantic_retrieval.laya_rerank.evidence import decode_evidence
from laya.vlm import build_vlm_inputs,collate_vlm,QTYPES
from laya.common import truncation_answer
r=Path.cwd();c=r/'experiments/laya-rerank-validation-v1';fixed=json.loads((c/'fixed.json').read_text());data=cohort(r,fixed);root=Path(json.loads((c/'study.yaml').read_text())['accuracy_result']['artifact_path']);report=json.loads((root/'provenance/laya-validation.json').read_text());numeric=load_array(root,report['arrays']['logits_probability']);ranked=load_array(root,report['arrays']['reranked_order']);targets=load_array(root,report['arrays']['targets']);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
judge=LayaJudge(fixed['weights'],device='cpu');memory=read_visual(r,root,data['rows']);cases=[];seen=set()
for i in range(10000):
 if data['order'][i,0]==targets[i] and ranked[i,0]!=targets[i] and targets[i] not in seen:
  seen.add(targets[i]);cases.append(i)
 if len(cases)==4:break
print('selected_case_queries',[(i,data['queries'][i]['query_text']) for i in cases],flush=True)
first=cases[0];ids=data['order'][first,:8];qs=[data['queries'][first]['query_text']]*8;ev=[memory[data['rows'][v]['video_id']] for v in ids];z8,p8=judge.score(qs,ev);zs=np.concatenate([judge.score([q],[e])[0] for q,e in zip(qs,ev)]);perm=[7,3,1,6,0,5,4,2];zr,_=judge.score([qs[i] for i in perm],[ev[i] for i in perm]);result=dict(batch8_vs_single_max=float(abs(z8-zs).max()),batch_permutation_max=float(abs(zr-z8[perm]).max()),cpu_vs_archived_gpu_logits_max=float(abs(z8-numeric[first,:8,:2]).max()),cases=[])
for i in cases:
 q=data['queries'][i]['query_text'];case=dict(query_index=i,query_id=data['queries'][i]['query_id'],query=q,candidates=[])
 for role,vi in [('gt',int(targets[i])),('winner',int(ranked[i,0]))]:
  row=data['rows'][vi];images,indices=decode_evidence(row);cached=memory[row['video_id']];z,p=judge.score([q],[cached]);fresh=judge.encode(images);zf,pf=judge.score([q],[fresh]);raw={};native=judge.agent.predict({**STATE,'images':images},{'check':{'type':'choice','instructions':QUESTION.format(q),'criteria':CRITERIA}},n_permutations=1,batch_size=1,prefix_cache=False,strict=True,_raw_logits=raw)
  qq=judge.agent._to_internal({'type':'choice','instructions':QUESTION.format(q),'criteria':CRITERIA});inp=build_vlm_inputs(judge.agent.processor,{**STATE,'images':images},qq,judge.agent.cfg['max_len'],judge.agent.cfg['head_max_len'],option_order=[0,1]);inp['qtype']=QTYPES['choice'];batch=collate_vlm([inp],judge.agent.processor.tokenizer.pad_token_id,with_pixels=True)
  with torch.inference_mode():pixels,_=judge.agent.model(*[batch[k] for k in ['input_ids','attention_mask','marker_pos','marker_mask','qtype']],pixel_values=batch['pixel_values'],pixel_attention_mask=batch['pixel_attention_mask'],option_span=batch['option_span'])
  pos=int(np.where(data['order'][i,:10]==vi)[0][0]);item=dict(role=role,video_id=row['video_id'],video_index=vi,frame_indices=indices,image_sizes=[list(im.size) for im in images],archived_p_match=float(numeric[i,pos,2]),cpu_cached_p_match=float(p[0]),fresh_p_match=float(pf[0]),native_p_match=native['answers']['check']['probabilities']['match'],native_cached_logits_max=float(abs(raw['check']-z[0]).max()),pixel_cached_logits_max=float(abs(pixels.numpy()-z).max()),fresh_cached_logits_max=float(abs(zf-z).max()),feature_cpu_gpu_max=float(abs(fresh['features'].numpy()-cached['features'].numpy()).max()),image_token_count=inp['ids'].count(judge.agent.processor.tokenizer.convert_tokens_to_ids(judge.agent.processor.image_token)),prefix_token_count=len(cached['ids']),sequence_token_count=len(inp['ids']),feature_shape=list(cached['features'].shape),truncation=truncation_answer(inp['truncation'],qq),native_choice=native['answers']['check']['choice'])
  case['candidates'].append(item);print('checked',i,role,item['native_cached_logits_max'],flush=True)
 result['cases'].append(case)
print(json.dumps(result,indent=2),flush=True);Path('/tmp/laya-real-cpu-diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
