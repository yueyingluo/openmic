"""Fit/select on development only; then report locked held-out comparisons."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from graded_text import metrics, read_lines, write_json
from omni_judge import DIMENSIONS
from analysis import clustered_bootstrap


def weight_fit(x,y):
    result=minimize(lambda w:np.mean((x@w-y)**2),np.repeat(.25,4),method='SLSQP',
                    bounds=[(0,1)]*4,constraints={'type':'eq','fun':lambda w:w.sum()-1},
                    options={'maxiter':1000,'ftol':1e-10})
    if not result.success or abs(result.x.sum()-1)>1e-6 or min(result.x)<-1e-6:
        raise ValueError('Invalid fitted weights')
    return np.maximum(result.x,0)/np.maximum(result.x,0).sum()


def assemble(references,predictions):
    refs={r['sample_id']:r for r in references}
    if len(refs)!=len(references):raise ValueError('Duplicate reference identities')
    pred=defaultdict(dict)
    for r in predictions:
        method=r['model_id'];sid=r['sample_id']
        if sid not in refs:raise ValueError('Prediction outside manifest')
        if sid in pred[method]:raise ValueError('Duplicate predictions')
        if r['score'] is not None and np.isfinite(r['score']):pred[method][sid]=r['score']
    return refs,dict(pred)


def select(refs,pred,mode,output):
    if output.exists():raise FileExistsError('Selection lock cannot be overwritten')
    methods=['direct','anchored'] if mode=='text' else ['audio_direct','audio_rubric']
    required=methods+DIMENSIONS if mode=='text' else methods
    ids=sorted(i for i,r in refs.items() if r['split']=='validation' and all(i in pred[m] for m in required))
    if len(ids)<20:raise ValueError('Too few complete validation items')
    y=np.array([refs[i]['target'] for i in ids])
    scores={m:metrics(y,[pred[m][i] for i in ids]) for m in methods}
    lock={'mode':mode,'validation_n':len(ids),'selection':'development Spearman; learned weights use group OOF predictions',
          'validation_metrics':scores,'project_human_alignment':False}
    if mode=='text':
        x=np.array([[pred[m][i] for m in DIMENSIONS] for i in ids])
        groups=[refs[i]['group'] for i in ids]
        oof=np.empty(len(ids))
        for train,valid in GroupKFold(n_splits=5).split(x,y,groups):
            oof[valid]=x[valid]@weight_fit(x[train],y[train])
        scores['dimensions_learned']=metrics(y,oof)
        lock['weights']=dict(zip(DIMENSIONS,weight_fit(x,y).tolist()))
        lock['equal_weights_validation']=metrics(y,x.mean(axis=1))
    chosen=max(scores,key=lambda m:scores[m]['spearman'] if scores[m]['spearman'] is not None else -2)
    lock['chosen_method']=chosen
    lock['validation_ids_sha256']=hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    write_json(output,lock)
    return lock


def report(refs,pred,lock,output,resamples=5000):
    pred={m:dict(p) for m,p in pred.items()}
    if lock['mode']=='text':
        ids=set.intersection(*(set(pred[m]) for m in DIMENSIONS))
        pred['dimensions_equal']={i:float(np.mean([pred[m][i] for m in DIMENSIONS])) for i in ids}
        pred['dimensions_learned']={i:sum(lock['weights'][m]*pred[m][i] for m in DIMENSIONS) for i in ids}
    planned=sorted(i for i,r in refs.items() if r['split']=='test')
    methods=list(pred)
    common=[i for i in planned if all(i in pred[m] for m in methods)]
    result={'scope':'public_label_alignment_not_project_human_validation','selection_lock':lock,
            'planned_test_n':len(planned),'common_test_n':len(common),'methods':{},'common_methods':{}}
    for m in methods:
        valid=[i for i in planned if i in pred[m]]
        result['methods'][m]=metrics([refs[i]['target'] for i in valid],[pred[m][i] for i in valid])
        result['methods'][m]['coverage']=len(valid)/len(planned) if planned else 0
        result['common_methods'][m]=metrics([refs[i]['target'] for i in common],[pred[m][i] for i in common])
        if m=='length':
            for key in ('methods','common_methods'):
                result[key][m]['mae']=result[key][m]['rmse']=None
    # Fixed shared resamples compare the locked primary judge with its relevant controls.
    controls=['ridge','ordinal_expected_multiclass','bert_five_class_ensemble','anchored','direct','dimensions_equal'] if lock['mode']=='text' else ['text_only','audio_direct','audio_rubric']
    selected=list(dict.fromkeys([lock['chosen_method']]+[m for m in controls if m in pred]))
    common_boot=[i for i in planned if all(i in pred[m] for m in selected)]
    result['bootstrap']=clustered_bootstrap(common_boot,{i:{'topic_id':refs[i]['group']} for i in common_boot},
        {i:refs[i]['target'] for i in common_boot},{m:pred[m] for m in selected},n=resamples,seed=20261026)
    result['bootstrap']['common_n']=len(common_boot)
    write_json(output,result)
    print(json.dumps({'chosen':lock['chosen_method'],'test':result['common_methods']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--references',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,nargs='+',required=True)
    parser.add_argument('--mode',choices=['text','audio'],required=True)
    parser.add_argument('--lock',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--select-only',action='store_true')
    parser.add_argument('--resamples',type=int,default=5000)
    args=parser.parse_args()
    refs,pred=assemble(read_lines(args.references),sum([read_lines(p) for p in args.predictions],[]))
    if args.select_only:
        select(refs,pred,args.mode,args.lock)
    else:
        if args.output is None:parser.error('--output required for report')
        report(refs,pred,json.loads(args.lock.read_text()),args.output,args.resamples)
