"""Paired source-domain uncertainty check; not a human-ranking experiment."""
from collections import defaultdict
import numpy as np
from common import ROOT, read_json, read_jsonl, write_json, sha256


def f1(y,p):
    pred=p>=.5
    a=[]
    for k in [0,1]:
        tp=np.sum((y==k)&(pred==k)); fp=np.sum((y!=k)&(pred==k)); fn=np.sum((y==k)&(pred!=k))
        a.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return float(np.mean(a))


def main():
    rows=[r for r in read_jsonl(ROOT/'data/recognition.jsonl') if r['split']=='source_holdout']
    ids=[r['sample_id'] for r in rows]; y=np.array([r['label'] for r in rows])
    bert=np.mean([read_json(ROOT/f'gpu_results/bert_seed_{s}_predictions.json')['source_holdout'] for s in [17,29,43]],axis=0)
    baseline={r['sample_id']:r['score'] for r in read_jsonl(ROOT/'data/source_predictions.jsonl')
              if r['split']=='source_holdout' and r['model_id']=='char_tfidf_logreg'}
    assert set(ids)==set(baseline)
    lr=np.array([baseline[i] for i in ids]); groups=defaultdict(list)
    for i,r in enumerate(rows): groups[r['duplicate_group']].append(i)
    clusters=list(groups.values()); rng=np.random.default_rng(20261028); draws=[]
    for _ in range(5000):
        chosen=np.concatenate([clusters[i] for i in rng.integers(0,len(clusters),size=len(clusters))])
        draws.append((f1(y[chosen],bert[chosen]),f1(y[chosen],lr[chosen])))
    draws=np.asarray(draws)
    report={'scope':'exploratory_source_classification_comparison_not_human_alignment',
        'sample_n':len(rows),'cluster_n':len(clusters),'cluster_unit':'duplicate_component','bootstrap_seed':20261028,'resamples':5000,
        'bert_ensemble_macro_f1':f1(y,bert),'logreg_macro_f1':f1(y,lr),
        'bert_ci95':np.quantile(draws[:,0],[.025,.975]).tolist(),'logreg_ci95':np.quantile(draws[:,1],[.025,.975]).tolist(),
        'paired_f1_difference':f1(y,bert)-f1(y,lr),'paired_difference_ci95':np.quantile(draws[:,0]-draws[:,1],[.025,.975]).tolist(),
        'manifest_sha256':sha256(ROOT/'data/recognition.jsonl'),
        'limitations':['Source-domain task only','Shared-source/paraphrase dependence may remain beyond duplicate components','Single fixed train/validation/holdout split; intervals do not include resplitting or retraining uncertainty']}
    write_json(ROOT/'reports/bert_source_comparison.json',report)
    print(report)


if __name__=='__main__': main()
