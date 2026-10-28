"""Posthoc read-only statistics; no inference, configuration selection or new ranking arm."""
import json
from pathlib import Path
import sys
import re
import numpy as np
repo=Path.cwd();sys.path[:0]=[str(repo),str(repo/'src')]
from semantic_retrieval.laya_rerank.science import cohort,load_array
control=repo/'experiments/laya-rerank-validation-v1'
data=cohort(repo,json.loads((control/'fixed.json').read_text()))
root=Path(json.loads((control/'study.yaml').read_text())['accuracy_result']['artifact_path'])
report=json.loads((root/'provenance/laya-validation.json').read_text())
a={k:load_array(root,v) for k,v in report['arrays'].items()}
o,b,t,z=a['baseline_order'],a['reranked_order'],a['targets'],a['logits_probability']
p=z[:,:,2];hit=o[:,:10]==t[:,None];eligible=hit.any(1);pos=hit.argmax(1);gt=p[np.arange(len(t)),pos]
validneg=(~hit)&eligible[:,None];pairs=(gt[:,None]>p)[validneg];ties=(gt[:,None]==p)[validneg]
good=o[:,0]==t;new=b[:,0]==t
quant=lambda x:np.quantile(x,[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist()
stats=dict(n=len(t),eligible=int(eligible.sum()),baseline_r1_conditional=float(good[eligible].mean()),laya_r1_conditional=float(new[eligible].mean()),uniform_top10_global_r1=float(eligible.mean()/10),retained=int((good&new).sum()),retention_fraction=float(new[good].mean()),rescued=int((~good&new).sum()),eligible_wrong_rescue_fraction=float(new[eligible&~good].mean()),conditional_pairwise_win_rate=float((pairs.astype(float)+.5*ties).mean()),all_p_quantiles=quant(p),gt_p_quantiles=quant(gt[eligible]),negative_p_quantiles=quant(p[~hit]),all_match_fraction=float((p>.5).mean()),gt_match_fraction=float((gt[eligible]>.5).mean()),negative_match_fraction=float((p[~hit]>.5).mean()),exact_tied_top_queries=int(((p==p.max(1,keepdims=True)).sum(1)>1).sum()),winner_original_position_counts=np.bincount(p.argmax(1),minlength=10).tolist(),gt_reranked_position_counts=np.bincount(np.argmax(b[eligible,:10]==t[eligible,None],axis=1),minlength=10).tolist(),within_query_p_std_quantiles=quant(p.std(1)),logit_margin_quantiles=quant(z[:,:,0]-z[:,:,1]))
length=np.array([len(q['query_text'].split()) for q in data['queries']])
temporal=np.array([bool(re.search(r'\b(before|after|then|while|starts?|ends?|turns?|changes?)\b',q['query_text'].lower())) for q in data['queries']])
groups=[('all',np.ones(len(t),dtype=bool)),('words<=7',length<=7),('words8-11',(length>=8)&(length<=11)),('words12-17',(length>=12)&(length<=17)),('words>=18',length>=18),('temporal_heuristic',temporal),('other',~temporal)]
sub=[dict(group=name,n=int(mask.sum()),baseline_r1=float(100*good[mask].mean()),laya_r1=float(100*new[mask].mean()),retained_fraction=float(new[mask&good].mean())) for name,mask in groups]
win=b[:,0];counts=np.bincount(win,minlength=500);basecounts=np.bincount(o[:,0],minlength=500);exposures=np.bincount(o[:,:10].ravel(),minlength=500)
hubs=[dict(video_id=data['rows'][i]['video_id'],winner_count=int(counts[i]),baseline_winner_count=int(basecounts[i]),gt_queries=20,correct_wins=int(((win==i)&new).sum()),top10_exposures=int(exposures[i]),win_fraction_when_exposed=float(counts[i]/exposures[i])) for i in np.argsort(-counts)[:5]]
reverse=float(100*(o[np.arange(len(t)),p.argmin(1)]==t).mean())
additional=dict(subgroups=sub,baseline_conditional_pairwise_win_rate=float(((9-pos[eligible])/9).mean()),changed_winners=int((b[:,0]!=o[:,0]).sum()),top_winner_hubs=hubs,reverse_option_mapping_diagnostic_r1=reverse)
gap=np.sort(p,axis=1)[:,-1]-np.sort(p,axis=1)[:,-2]
margin=dict(eligible_negative_median=float(np.median(p[validneg])),winner_margin_quantiles=np.quantile(gap,[0,.01,.1,.5,.9,.99,1]).tolist(),queries_with_margin_le_2e5=int((gap<=2e-5).sum()),reversed_mapping_global_r1=reverse)
output={'stats':stats,'additional-stats':additional,'margin-stats':margin}
Path('/tmp/laya-failure-statistics-recomputed.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
