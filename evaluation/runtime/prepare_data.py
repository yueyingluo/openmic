"""Prepare the pinned public recognition task, without fitting on held-out data."""
import collections
import json
import re
import unicodedata
import numpy as np
import yaml
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tokenizers import BertWordPieceTokenizer
from common import ROOT, DESIGN, sha256, write_json, write_jsonl


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()


def trigrams(text):
    return sorted({text[i:i+3] for i in range(max(1, len(text)-2))})


def near_components(texts, threshold=0.85):
    """Exact Jaccard comparisons from sparse intersections; no approximate LSH."""
    parent = list(range(len(texts)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    if not texts:
        return [], []
    m = CountVectorizer(analyzer=trigrams, binary=True, dtype=np.int32).fit_transform(texts)
    sizes = np.asarray(m.sum(axis=1)).ravel()
    edges = []
    for start in range(0, len(texts), 128):
        overlap = (m[start:start+128] @ m.T).tocoo()
        for r, j, n in zip(overlap.row, overlap.col, overlap.data):
            i = start + int(r)
            j = int(j)
            if j <= i:
                continue
            union = int(sizes[i] + sizes[j] - n)
            if union and n / union >= threshold:
                a, b = find(i), find(j)
                parent[max(a, b)] = min(a, b)
                edges.append((i, j, float(n / union)))
    return [find(i) for i in range(len(texts))], edges


def main():
    cfg = yaml.safe_load((DESIGN/'experiment.yaml').read_text())
    dc = cfg['data']['public_binary']
    raw = ROOT/'data/raw/CFunSet.json'
    digest = sha256(raw)
    if digest != dc['expected_sha256']:
        raise ValueError('Source hash mismatch: refusing to prepare data')
    data = json.loads(raw.read_bytes())
    selected = [(i, r) for i, r in enumerate(data) if r['instruction'].startswith(dc['instruction_prefix'])]
    instructions = collections.Counter(r['instruction'] for _, r in selected)
    if len(instructions) != 1:
        raise ValueError('Recognition instruction changed; review before use')
    labels = dc['allowed_labels']
    actual = collections.Counter(r['output'] for _, r in selected)
    if set(actual) != set(labels):
        raise ValueError(f'Unexpected labels: {actual}')
    tokenizer_cfg = json.loads((ROOT/'data/raw/bert_tokenizer_config.json').read_text())
    tok = BertWordPieceTokenizer(str(ROOT/'data/raw/bert_vocab.txt'), lowercase=tokenizer_cfg.get('do_lower_case', True))
    exact = collections.defaultdict(list)
    quarantine = []
    for index, row in selected:
        text = row['input']
        if not isinstance(text, str) or not normalize(text):
            quarantine.append({'source_index': index, 'reason': 'empty_input'})
            continue
        exact[normalize(text)].append({'source_index': index, 'text': text, 'label': labels[row['output']]})
    representatives = []
    for norm, members in sorted(exact.items(), key=lambda kv: min(r['source_index'] for r in kv[1])):
        if len({r['label'] for r in members}) > 1:
            quarantine.extend(dict(r, reason='conflicting_exact_labels') for r in members)
            continue
        first = members[0]
        n_tokens = len(tok.encode(first['text']).ids)  # Includes CLS and SEP, never truncates.
        if n_tokens > cfg['bert']['max_length']:
            quarantine.extend(dict(r, reason='over_512_bert_tokens', tokens=n_tokens) for r in members)
            continue
        representatives.append(dict(first, sample_id=f'CF-{first["source_index"]:06d}',
                                    source_indices=[r['source_index'] for r in members],
                                    text_sha256=sha256(first['text']), normalized_sha256=sha256(norm),
                                    bert_tokens=n_tokens, normalized=norm))
    roots, edges = near_components([r['normalized'] for r in representatives], dc['near_duplicate_threshold'])
    components = collections.defaultdict(list)
    for root, row in zip(roots, representatives):
        components[root].append(row)
    retained = {}
    for root, members in components.items():
        if len({r['label'] for r in members}) > 1:
            quarantine.extend(dict(r, reason='conflicting_near_duplicate_component') for r in members)
        else:
            retained[root] = members
    groups = sorted(retained)
    group_labels = [retained[g][0]['label'] for g in groups]
    ratios = dc['fallback_split_ratios']
    train_g, rest_g = train_test_split(groups, test_size=ratios[1]+ratios[2], stratify=group_labels, random_state=dc['split_seed'])
    val_g, test_g = train_test_split(rest_g, test_size=ratios[2]/sum(ratios[1:]),
                                    stratify=[retained[g][0]['label'] for g in rest_g], random_state=dc['split_seed']+1)
    outputs = []
    for split, ids in [('train', train_g), ('validation', val_g), ('source_holdout', test_g)]:
        for g in sorted(ids):
            for item in retained[g]:
                row = {k:v for k,v in item.items() if k != 'normalized'}
                outputs.append(dict(row, split=split, duplicate_group=f'G-{g:05d}'))
    write_jsonl(ROOT/'data/recognition.jsonl', outputs)
    write_jsonl(ROOT/'data/quarantine.jsonl', quarantine)
    write_jsonl(ROOT/'data/near_duplicate_edges.jsonl', [dict(left=representatives[a]['sample_id'], right=representatives[b]['sample_id'], jaccard=s) for a,b,s in edges])
    assert len({r['sample_id'] for r in outputs}) == len(outputs)
    group_splits = collections.defaultdict(set)
    for r in outputs:
        group_splits[r['duplicate_group']].add(r['split'])
    assert all(len(s)==1 for s in group_splits.values())
    report = {
        'source': dc['repo_id'], 'revision': dc['revision'], 'sha256': digest,
        'raw_records': len(data), 'recognition_records': len(selected), 'label_counts_raw': dict(actual),
        'recognition_instruction': next(iter(instructions)), 'label_mapping': labels,
        'label_audit': 'Exact output strings and task definition verified; semantic label correctness not independently human-audited.',
        'distinct_normalized_inputs': len(exact), 'extra_exact_copies': sum(len(v)-1 for v in exact.values()),
        'quarantine_by_reason': dict(collections.Counter(r['reason'] for r in quarantine)),
        'near_duplicate_edges': len(edges), 'retained_components': len(retained), 'retained_records': len(outputs),
        'split_counts': {s:dict(collections.Counter(r['label'] for r in outputs if r['split']==s)) for s in ['train','validation','source_holdout']},
        'split_policy': 'No official held-out split in source JSON. Fixed-seed component-stratified local split; exact copies collapsed to first representative. Mixed-label near components quarantined.',
        'cross_split_duplicate_groups': 0, 'manifest_sha256': sha256(ROOT/'data/recognition.jsonl'),
        'tokenizer': {'implementation': 'tokenizers.BertWordPieceTokenizer', 'vocab_sha256': sha256(ROOT/'data/raw/bert_vocab.txt'), 'config': tokenizer_cfg, 'special_tokens_included': True, 'truncation': False},
        'limitations': ['Source labels assess recognition, not 1–5 funniness.', 'Not an unseen benchmark for CFunModel.', 'Trigram deduplication does not exclude paraphrases or shared original sources.'],
    }
    write_json(ROOT/'reports/data_audit.json', report)
    print(json.dumps({k:report[k] for k in ['recognition_records','extra_exact_copies','quarantine_by_reason','retained_records','split_counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
