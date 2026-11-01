import csv
import json

import pytest

from audit_graded_sources import audit


def write_table(root, name, rows, delimiter=','):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def corpora(tmp_path):
    train = [{'joke': 'Shared!', 'label': '1'}, {'joke': 'middle', 'label': '3'},
             {'joke': 'high', 'label': '5'}]
    test = [{'joke': '<b>ＳＨＡＲＥＤ</b>', 'label': '1'},
            {'joke': 'other middle', 'label': '3'}, {'joke': 'other high', 'label': '5'}]
    write_table(tmp_path, 'ccl2019/task2_train.csv', train)
    write_table(tmp_path, 'ccl2019/task2_test_with_label.csv', test)
    write_table(tmp_path, 'ccl2019/task2_development.csv', [{'joke': 'shared'}])
    cmhl = [{'ID': str(min(i, 4)), 'Content': f'joke number {i}', 'HumorLevel': str(i)}
            for i in range(1, 6)]
    for name in ['JokeHumorLevel', 'JokeSkill', 'JokeMotivation']:
        write_table(tmp_path, f'cmhl/{name}.txt', cmhl, '\t')
    return tmp_path


def test_overlap_and_reused_ids_without_exposing_text(corpora):
    cfun = corpora / 'cfun.json'
    cfun.write_text(json.dumps([{'input': 'shared', 'output': 'joke number 2'},
                               {'input': 'SHARED!', 'output': ''}]))
    result = audit(corpora, cfun)
    assert result['split_overlap']['ccl_train_test_exact'] == 1
    assert result['split_overlap']['ccl_dev_test_exact'] == 1
    assert result['cmhl_reused_id_rows'] == 1
    assert result['cfunset_exact_overlap_lower_bound']['cmhl/JokeHumorLevel.txt'] == 1
    assert result['cfunset_exact_overlap_lower_bound']['ccl2019/task2_train.csv'] == 1
    assert result['files']['ccl2019/task2_development.csv']['label_counts'] is None
    assert 'joke number' not in json.dumps(result)


def test_annotation_misalignment_is_rejected(corpora):
    path = corpora / 'cmhl/JokeSkill.txt'
    path.write_text(path.read_text().replace('joke number 1', 'wrong story'))
    with pytest.raises(ValueError, match='misaligned'):
        audit(corpora)


def test_unexpected_label_scale_is_rejected(corpora):
    write_table(corpora, 'ccl2019/task2_train.csv', [{'joke': 'binary', 'label': '0'}])
    with pytest.raises(ValueError, match='Unexpected label'):
        audit(corpora)
