"""Five-level CMHL preparation and CPU baselines. No project human ratings read."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
from opencc import OpenCC
from scipy.stats import spearmanr, kendalltau
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics.pairwise import cosine_similarity

CONVERTER = OpenCC('t2s')


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def canonical(text, audit=False):
    if audit:
        text = html.unescape(re.sub(r'<[^>]+>', '', text))
    text = CONVERTER.convert(unicodedata.normalize('NFKC', text)).lower()
    if audit:
        return ''.join(c for c in text if c.isalnum())
    return ''.join(re.findall(r'[\u3400-\u9fffA-Za-z0-9]+', text))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def write_lines(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows))


def read_lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def assign_groups(records, threshold=.9):
    """Same label-independent grouping/hash rule as text PR1 at f8e61fc."""
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    matrix = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), dtype=np.float32).fit_transform(
        [r['canonical'] for r in records])
    sim = cosine_similarity(matrix)
    left, right = np.where(np.triu(sim, k=1) >= threshold)
    for a, b in zip(left, right):
        parent[find(b)] = find(a)
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    for members in groups.values():
        digest = sha(min(records[i]['canonical'] for i in members))
        bucket = int(digest[:8], 16) % 10
        for i in members:
            records[i]['group'] = digest
            records[i]['split'] = 'test' if bucket < 2 else 'validation' if bucket == 2 else 'train'
    return len(groups)


def prepare(corpus, cfun, output):
    output.mkdir(parents=True, exist_ok=False)
    rows = list(csv.DictReader((corpus/'JokeHumorLevel.txt').open(encoding='utf-8-sig'), delimiter='\t'))
    records = []
    for i, r in enumerate(rows):
        key = canonical(r['Content'])
        records.append({'sample_id':r['ID']+'-'+sha(key)[:10], 'source_row':i+2,
                        'text':CONVERTER.convert(' '.join(r['Content'].split())),
                        'target':int(r['HumorLevel']), 'canonical':key})
    if len({r['sample_id'] for r in records}) != len(records):
        raise ValueError('Duplicate identities: audit source before preparing')
    records.sort(key=lambda r:r['sample_id'])
    groups = assign_groups(records)
    original_counts = dict(Counter(r['split'] for r in records))
    source = json.loads(cfun.read_text())
    raw = {r[k] for r in source for k in ('input', 'output') if r.get(k)}
    print('Normalizing CFunSet contamination reference', len(raw), flush=True)
    cfun_texts = {canonical(s, audit=True) for s in raw}
    excluded_groups = {r['group'] for r in records if canonical(r['text'], audit=True) in cfun_texts}
    retained = [r for r in records if r['group'] not in excluded_groups]
    for r in retained:
        r.pop('canonical')
    # Select by identity only, before model predictions; no score balancing or cherry-picking.
    selected = []
    for split, limit in [('validation',96), ('test',192)]:
        pool = sorted((r for r in retained if r['split']==split), key=lambda r:sha('omni-v2-'+r['sample_id']))
        selected.extend(pool[:limit])
    write_lines(output/'records.private.jsonl', retained)
    write_lines(output/'judge_inputs.jsonl', [{k:r[k] for k in ('sample_id','text','group','split')} for r in selected])
    write_json(output/'preparation.json', {
        'source':'CMHL', 'source_revision':'d6ce23497ac9bc50071600c65b465b7e14a861df',
        'grouping_reference':'text PR1 f8e61fcaea0567ce5ab143d9d1b0c0e38631abb6',
        'original_split_counts':original_counts, 'original_groups':groups,
        'excluded_cfun_groups':len(excluded_groups), 'retained_rows':len(retained),
        'split_counts':dict(Counter(r['split'] for r in retained)),
        'labels_by_split':{s:dict(sorted(Counter(r['target'] for r in retained if r['split']==s).items()))
                           for s in ('train','validation','test')},
        'judge_subset':dict(Counter(r['split'] for r in selected)),
        'selection_rule':'first by sha256(omni-v2- + sample_id), without labels',
        'input_sha256':hashlib.sha256((output/'judge_inputs.jsonl').read_bytes()).hexdigest(),
        'cfunset_sha256':hashlib.sha256(cfun.read_bytes()).hexdigest(),
        'remaining_pretraining_or_semantic_overlap_not_excluded':True,
        'project_human_alignment':False})
    return retained


def metrics(target, prediction):
    y = np.asarray(target, dtype=float)
    p = np.asarray(prediction, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    variable = len(y)>=3 and len(set(y))>1 and len(set(p))>1
    return {'n':len(y), 'coverage':float(valid.mean()) if len(valid) else 0,
            'spearman':float(spearmanr(y,p).statistic) if variable else None,
            'kendall_tau_b':float(kendalltau(y,p).statistic) if variable else None,
            'mae':float(np.abs(y-p).mean()) if len(y) else None,
            'rmse':float(np.sqrt(((y-p)**2).mean())) if len(y) else None}


def train(root):
    records = read_lines(root/'records.private.jsonl')
    parts = {s:[r for r in records if r['split']==s] for s in ('train','validation','test')}
    vec = TfidfVectorizer(analyzer='char', ngram_range=(2,5), min_df=2, sublinear_tf=True, max_features=150000)
    x = {'train':vec.fit_transform([r['text'] for r in parts['train']])}
    x.update({s:vec.transform([r['text'] for r in parts[s]]) for s in ('validation','test')})
    y = {s:np.array([r['target'] for r in rows]) for s,rows in parts.items()}
    candidates = []
    for alpha in [.1,1,10]:
        model = Ridge(alpha=alpha).fit(x['train'],y['train'])
        score = metrics(y['validation'],np.clip(model.predict(x['validation']),1,5))
        candidates.append(('ridge',alpha,model,score))
    for c in [.1,1,10]:
        model = LogisticRegression(C=c,max_iter=1000,random_state=20261026).fit(x['train'],y['train'])
        score = metrics(y['validation'],model.predict_proba(x['validation'])@model.classes_)
        candidates.append(('ordinal_expected_multiclass',c,model,score))
    chosen = {name:max([t for t in candidates if t[0]==name],key=lambda t:t[3]['spearman'] if t[3]['spearman'] is not None else -2)
              for name in ('ridge','ordinal_expected_multiclass')}
    predictions = []
    for s in ('validation','test'):
        methods = {'train_mean':np.repeat(y['train'].mean(),len(y[s])), 'length':np.array([len(r['text']) for r in parts[s]])}
        for name,(_,_,model,_) in chosen.items():
            methods[name] = np.clip(model.predict(x[s]),1,5) if name=='ridge' else model.predict_proba(x[s])@model.classes_
        for name,p in methods.items():
            for r,v in zip(parts[s],p):
                predictions.append({'sample_id':r['sample_id'],'split':s,'model_id':name,'score':float(v)})
    write_lines(root/'cpu_predictions.private.jsonl',predictions)
    report = {'validation_candidates':[{'method':n,'parameter':p,'metrics':m} for n,p,_,m in candidates],
              'chosen_parameters':{n:t[1] for n,t in chosen.items()}, 'full_test':{},
              'meaning':'agreement_with_single_annotator_public_labels_not_project_human_alignment'}
    for name in ('train_mean','length',*chosen):
        p = [r['score'] for r in predictions if r['model_id']==name and r['split']=='test']
        report['full_test'][name] = metrics(y['test'],p)
        if name=='length':
            report['full_test'][name]['mae']=report['full_test'][name]['rmse']=None
    write_json(root/'cpu_metrics.json',report)
    print(json.dumps(report['full_test'],ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus',type=Path)
    parser.add_argument('--cfunset',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-only',action='store_true')
    args = parser.parse_args()
    if not args.train_only:
        if not args.corpus or not args.cfunset:
            parser.error('--corpus and --cfunset required for preparation')
        prepare(args.corpus,args.cfunset,args.output)
    train(args.output)
