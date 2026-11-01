import json
from pathlib import Path
import numpy as np
import pytest
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from common import DESIGN
from label_likelihood import candidate_suffix


def test_label_prefix_and_end_of_turn_boundary():
    assert candidate_suffix([10,11],[10,11,20,21,99,5],99)==[20,21,99]
    with pytest.raises(ValueError,match='prefix'):
        candidate_suffix([10,11],[10,12,20,99],99)
    with pytest.raises(ValueError,match='end-of-turn'):
        candidate_suffix([10,11],[10,11,20],99)
    with pytest.raises(ValueError,match='Empty'):
        candidate_suffix([10],[10,99],99)


def test_reported_metrics_match_confusion_matrices():
    records=json.loads((DESIGN/'results/label_likelihood_metrics.json').read_text())
    for record in records:
        metrics=record['source_metrics']
        matrix=metrics['confusion_matrix_labels_0_1']
        y=[];pred=[]
        for actual in [0,1]:
            for predicted in [0,1]:
                y.extend([actual]*matrix[actual][predicted])
                pred.extend([predicted]*matrix[actual][predicted])
        assert len(y)==metrics['n']==628
        assert f1_score(y,pred,average='macro')==pytest.approx(metrics['macro_f1'])
        assert balanced_accuracy_score(y,pred)==pytest.approx(metrics['balanced_accuracy'])
        assert accuracy_score(y,pred)==pytest.approx(metrics['accuracy'])
        assert record['human_alignment_validated'] is False


def test_source_counts_and_first_pass_coverage():
    audit=json.loads((DESIGN/'results/data_audit.json').read_text())
    assert audit['recognition_records']==4200
    assert audit['retained_records']==4186
    assert sum(sum(counts.values()) for counts in audit['split_counts'].values())==4186
    records=json.loads((DESIGN/'results/llm_first_pass_verification.json').read_text())
    assert len(records)==2
    assert all(r['scored_responses']==7 and r['samples']==12 for r in records)


def test_gpu_metric_calculation_matches_reference():
    from gpu_experiment import binary_metrics
    from baseline import metrics
    y=[0,1,1,0,1,0,1,0]
    p=[.1,.8,.4,.6,.5,.5,.9,.2]
    got=binary_metrics(y,p);expected=metrics(y,p)
    for key in ['macro_f1','balanced_accuracy','accuracy','roc_auc']:
        assert got[key]==pytest.approx(expected[key])
    assert got['confusion_matrix_labels_0_1']==expected['confusion_matrix_labels_0_1']


def test_gpu_recheck_rejects_another_job_without_loading_cuda(monkeypatch):
    import os
    import gpu_experiment
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-test-device')
    monkeypatch.setattr(gpu_experiment.subprocess,'check_output',lambda *a,**k:'GPU-test-device, 999999999\n')
    with pytest.raises(RuntimeError,match='another compute process'):
        gpu_experiment.assert_gpu_available()
    monkeypatch.setattr(gpu_experiment.subprocess,'check_output',lambda *a,**k:'GPU-test-device, {}\n'.format(os.getpid()))
    gpu_experiment.assert_gpu_available()
