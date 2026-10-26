"""Audit locally downloaded graded corpora; emit counts/hashes, never source text.

Input files must match the pinned manifests recorded in e2e_sources.json.
This command does not download, train, choose prompts, or score test examples.
"""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import unicodedata


def read_table(path, delimiter=','):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def audit(root, cfun_path=None, simplify=False):
    converter=None
    if simplify:
        from opencc import OpenCC
        converter=OpenCC('t2s')
    def canonical(text):
        text=unicodedata.normalize('NFKC', html.unescape(re.sub(r'<[^>]+>', '', text))).lower()
        if converter:
            text=converter.convert(text)
        return ''.join(c for c in text if c.isalnum())
    tables=[('ccl2019/task2_train.csv','joke','label',','),
            ('ccl2019/task2_development.csv','joke',None,','),
            ('ccl2019/task2_test_with_label.csv','joke','label',','),
            ('cmhl/JokeHumorLevel.txt','Content','HumorLevel','\t')]
    report={'scope':'source_data_audit_not_model_evaluation','files':{},'split_overlap':{}}
    sets={}
    for file,field,label,delimiter in tables:
        path=root/file
        rows=read_table(path,delimiter)
        texts=[canonical(r[field]) for r in rows]
        sets[file]=set(texts)
        labels=Counter(r[label] for r in rows) if label else None
        expected={'1','3','5'} if file.startswith('ccl2019') else {'1','2','3','4','5'}
        if labels is not None and set(labels)!=expected:
            raise ValueError('Unexpected label set in '+file)
        report['files'][file]={'rows':len(rows),'label_counts':dict(sorted(labels.items())) if labels else None,
            'unique_normalized_texts':len(set(texts)),'empty_texts':texts.count(''),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    cmhl=read_table(root/'cmhl/JokeHumorLevel.txt','\t')
    for name in ['JokeSkill','JokeMotivation']:
        rows=read_table(root/'cmhl'/f'{name}.txt','\t')
        if len(rows)!=len(cmhl) or any(a['ID']!=b['ID'] or canonical(a['Content'])!=canonical(b['Content']) for a,b in zip(cmhl,rows)):
            raise ValueError('CMHL annotation tables are misaligned')
    report['cmhl_reused_id_rows']=len(cmhl)-len({r['ID'] for r in cmhl})
    train=sets[tables[0][0]];dev=sets[tables[1][0]];test=sets[tables[2][0]]
    report['split_overlap']={'ccl_train_test_exact':len(train&test),'ccl_dev_test_exact':len(dev&test),
        'ccl_dev_train_exact':len(dev&train),'cmhl_ccl_train_exact':len(sets[tables[3][0]]&train),
        'cmhl_ccl_test_exact':len(sets[tables[3][0]]&test)}
    if cfun_path:
        raw=json.loads(cfun_path.read_text())
        source_texts={r[k] for r in raw for k in ['input','output'] if r.get(k)}
        cfun={canonical(s) for s in source_texts}
        report['cfunset_exact_overlap_lower_bound']={file:len(texts&cfun) for file,texts in sets.items()}
        report['cfunset_sha256']=hashlib.sha256(cfun_path.read_bytes()).hexdigest()
    report['normalization']={'html_unescape_and_tag_removal':True,'nfkc':True,'lower':True,
        'alphanumeric_only':True,'traditional_to_simplified':simplify,
        'fuzzy_or_substring_matching':False,'contamination_counts_are_lower_bounds':True}
    report['raw_texts_or_individual_scores_published']=False
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--cfunset',type=Path)
    parser.add_argument('--simplify',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.root,args.cfunset,args.simplify)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(json.dumps({'status':'audited','files':len(result['files']),'output':str(args.output)}))
