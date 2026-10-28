import sys,json
from pathlib import Path
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'src')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from semantic_retrieval.laya_rerank.science import cohort,load_array
from semantic_retrieval.laya_rerank.evidence import decode_evidence
r=Path.cwd();c=r/'experiments/laya-rerank-validation-v1';d=cohort(r,json.loads((c/'fixed.json').read_text()));a=Path(json.loads((c/'study.yaml').read_text())['accuracy_result']['artifact_path']);rep=json.loads((a/'provenance/laya-validation.json').read_text());z=load_array(a,rep['arrays']['logits_probability']);b=load_array(a,rep['arrays']['reranked_order']);t=load_array(a,rep['arrays']['targets']);cases=[0,20,40,66];fig,axes=plt.subplots(8,4,figsize=(13,18));texts=[]
for ci,i in enumerate(cases):
 texts.append(dict(query_index=i,query_id=d['queries'][i]['query_id'],query=d['queries'][i]['query_text'],candidates=[]))
 for ri,(role,vi) in enumerate([('GT',int(t[i])),('Laya winner',int(b[i,0]))]):
  v=d['rows'][vi];images,indices=decode_evidence(v);pos=int(np.where(d['order'][i,:10]==vi)[0][0]);prob=float(z[i,pos,2]);texts[-1]['candidates'].append(dict(role=role,video_id=v['video_id'],probability=prob,indices=indices,caption_examples=[q['query_text'] for q in d['queries'] if q['video_id']==v['video_id']][:3]))
  for col,im in enumerate(images):
   ax=axes[2*ci+ri,col];ax.imshow(im);ax.axis('off');ax.set_title(f'q{i} {role} | {v["video_id"]} | frame {indices[col]}'+(f' | p={prob:.3f}' if col==0 else ''),fontsize=8)
plt.tight_layout();fig.savefig('/tmp/laya-failure-cases.jpg',dpi=120)
Path('/tmp/laya-failure-cases.json').write_text(json.dumps(texts,indent=2));print(json.dumps(texts,indent=2))
