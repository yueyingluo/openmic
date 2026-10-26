"""Rank alignment on explicit human data. No guessed, imputed or zero-filled labels."""
import argparse
from collections import defaultdict
from itertools import combinations
import numpy as np
from scipy.stats import spearmanr, kendalltau
from common import finite_number, unique_by, read_jsonl, write_json


def correlations(human, model):
    if len(human) < 3:
        return {'spearman':None,'kendall_tau_b':None,'reason':'fewer_than_three_valid_samples'}
    if len(set(human))<2 or len(set(model))<2:
        return {'spearman':None,'kendall_tau_b':None,'reason':'constant_input'}
    return {'spearman':float(spearmanr(human,model).statistic),
            'kendall_tau_b':float(kendalltau(human,model,variant='b').statistic),'reason':None}


def ordinal_alpha(rating_lists):
    units = [r for r in rating_lists if len(r)>=2]
    if not units:
        return {'value':None,'reason':'fewer_than_two_raters_per_item'}
    counts=np.bincount([v for row in units for v in row],minlength=6)[1:6].astype(float)
    n=float(counts.sum())
    distance=np.zeros((5,5))
    for i,j in combinations(range(5),2):
        distance[i,j]=distance[j,i]=(sum(counts[i:j+1])-(counts[i]+counts[j])/2)**2
    observed=sum(sum(distance[a-1,b-1] for i,a in enumerate(row) for j,b in enumerate(row) if i!=j)/(len(row)-1) for row in units)/n
    expected=sum(counts[i]*counts[j]*distance[i,j] for i in range(5) for j in range(5))/(n*(n-1))
    return {'value':float(1-observed/expected) if expected else None,'reason':None if expected else 'no_rating_variation'}


def pair_accuracy(ids, samples, human, pred, threshold=.5):
    planned=valid=decisive=covered=correct=0
    for a,b in combinations(ids,2):
        if samples[a]['topic_id']!=samples[b]['topic_id']:
            continue
        planned+=1
        if a not in human or b not in human:
            continue
        valid+=1
        dh=human[a]-human[b]
        if abs(dh)<=threshold:
            continue
        decisive+=1
        if a not in pred or b not in pred:
            continue
        covered+=1
        correct+=int(dh*(pred[a]-pred[b])>0)  # Model tie is incorrect on decisive human pair.
    return {'planned_pairs':planned,'pairs_with_human_scores':valid,'human_decisive_pairs':decisive,
            'model_covered_decisive_pairs':covered,'correct':correct,
            'accuracy_on_covered_decisive':correct/covered if covered else None,
            'accuracy_missing_as_incorrect_on_decisive':correct/decisive if decisive else None,
            'covered_decisive_over_all_planned':covered/planned if planned else None}


def clustered_bootstrap(ids, samples, human, model_scores, n=5000, seed=20261026):
    clusters=defaultdict(list)
    for i in ids:
        clusters[samples[i]['topic_id']].append(i)
    keys=sorted(clusters)
    names=sorted(model_scores)
    if len(keys)<2:
        return {'reason':'fewer_than_two_topic_clusters','n_clusters':len(keys),'intervals':{},'paired_differences':{}}
    rng=np.random.default_rng(seed)
    values={name:[] for name in names}
    deltas={f'{a} minus {b}':[] for a,b in combinations(names,2)}
    for _ in range(n):
        draw=[i for k in rng.choice(keys,size=len(keys),replace=True) for i in clusters[k]]
        h=[human[i] for i in draw]
        current={name:correlations(h,[model_scores[name][i] for i in draw])['spearman'] for name in names}
        for name,v in current.items():
            if v is not None: values[name].append(v)
        for a,b in combinations(names,2):
            if current[a] is not None and current[b] is not None:
                deltas[f'{a} minus {b}'].append(current[a]-current[b])
    def interval(v):
        fraction=len(v)/n
        return {'valid_resamples':len(v),'valid_fraction':fraction,
                'ci95':np.quantile(v,[.025,.975]).tolist() if fraction>=.95 else None,
                'reason':None if fraction>=.95 else 'less_than_95_percent_valid_resamples'}
    return {'n_clusters':len(keys),'n_resamples':n,'seed':seed,'unit':'topic_cluster_fixed_rater_panel',
            'caution':'Few topics: interval may be unstable' if len(keys)<10 else None,
            'intervals':{k:interval(v) for k,v in values.items()},'paired_differences':{k:interval(v) for k,v in deltas.items()}}


def analyze(sample_rows, rating_rows, prediction_rows, split='pilot', min_raters=1, n_bootstrap=5000):
    if split=='test' and min_raters<3:
        raise ValueError('Formal test requires >=3 valid raters; use personal_test for a single-rater study')
    if n_bootstrap<1 or min_raters<1:
        raise ValueError('Invalid bootstrap count or rater threshold')
    all_samples={k[0]:r for k,r in unique_by(sample_rows,['sample_id']).items()}
    unique_by(rating_rows,['sample_id','rater_id','annotation_round'])
    unique_by(prediction_rows,['sample_id','model_id'])
    rounds={r['annotation_round'] for r in rating_rows}
    if len(rounds)>1:
        raise ValueError('Choose one annotation round; repeated ratings cannot be pooled as independent people')
    samples={i:r for i,r in all_samples.items() if r['split']==split}
    for i,r in samples.items():
        if not r.get('topic_id'):
            raise ValueError(f'Missing topic_id: {i}')
        if split in ('test','personal_test') and r.get('formal_test_eligible') is not True:
            raise ValueError('Development/pilot sample cannot enter a test analysis')
    grouped=defaultdict(dict)
    for r in rating_rows:
        if r['sample_id'] not in all_samples:
            raise ValueError('Unknown rating sample')
        score=r.get('humor_score')
        if score is not None and (not finite_number(score) or int(score)!=score or not 1<=score<=5):
            raise ValueError('Human score must be an integer 1–5 or null')
        if r.get('understood')=='no' and score is not None:
            raise ValueError('Ununderstood rating must be blank')
        if r['sample_id'] in samples and score is not None:
            grouped[r['sample_id']][r['rater_id']]=int(score)
    human={i:float(np.mean(list(rs.values()))) for i,rs in grouped.items() if len(rs)>=min_raters}
    model_scores=defaultdict(dict)
    for r in prediction_rows:
        if r['sample_id'] not in all_samples:
            raise ValueError('Unknown prediction sample')
        if r['sample_id'] not in samples: continue
        model_scores[r['model_id']]  # retain all-invalid models for coverage reporting
        if r.get('score') is not None:
            if not finite_number(r['score']): raise ValueError('Non-finite prediction')
            model_scores[r['model_id']][r['sample_id']]=float(r['score'])
    ids=sorted(samples)
    common=sorted(set(human).intersection(*(set(p) for p in model_scores.values()))) if model_scores else []
    report={'split':split,'inference_scope':'individual_preference_only' if len({p for rs in grouped.values() for p in rs})<=1 else 'observed_rater_panel_only',
            'planned_samples':len(samples),'valid_human_samples':len(human),'minimum_raters':min_raters,
            'common_valid_samples':len(common),'common_sample_ids':common,'models':{},
            'status':'computed' if human and model_scores else 'awaiting_human_ratings_or_predictions'}
    for name,p in model_scores.items():
        own=sorted(set(human)&set(p))
        report['models'][name]={'prediction_coverage':len(p)/len(samples) if samples else None,
            'own_valid_samples':len(own),'own_valid_set':correlations([human[i] for i in own],[p[i] for i in own]),
            'common_valid_set':correlations([human[i] for i in common],[p[i] for i in common]),
            'within_topic_pairs_by_human_tie_threshold':{str(t):pair_accuracy(ids,samples,human,p,t) for t in [0.0,0.5,1.0]}}
    report['bootstrap']=clustered_bootstrap(common,samples,human,model_scores,n_bootstrap)
    report['ordinal_krippendorff_alpha']=ordinal_alpha([list(v.values()) for v in grouped.values()])
    loo={}
    for rater in sorted({p for rs in grouped.values() for p in rs}):
        pairs=[(rs[rater],float(np.mean([v for k,v in rs.items() if k!=rater]))) for rs in grouped.values() if rater in rs and len(rs)>=2]
        loo[rater]=correlations([a for a,b in pairs],[b for a,b in pairs])
    report['leave_one_rater_out']=loo
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ['samples','ratings','predictions','output']: p.add_argument('--'+field,required=True)
    p.add_argument('--split',default='pilot')
    p.add_argument('--min-raters',type=int,default=1)
    args=p.parse_args()
    write_json(args.output,analyze(read_jsonl(args.samples),read_jsonl(args.ratings),read_jsonl(args.predictions),args.split,args.min_raters))
