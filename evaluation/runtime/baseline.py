"""CPU baselines with frozen hyperparameters. Source holdout is not target validation."""
import collections
import json
import warnings
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from common import ROOT, read_jsonl, write_json, write_jsonl, sha256


def metrics(y, p):
    pred = np.asarray(p) >= 0.5
    return {'n':len(y), 'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)), 'accuracy':float(accuracy_score(y,pred)),
            'roc_auc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
            'confusion_matrix_labels_0_1':confusion_matrix(y,pred,labels=[0,1]).tolist()}


def main():
    rows = read_jsonl(ROOT/'data/recognition.jsonl')
    train = [r for r in rows if r['split']=='train']
    y = [r['label'] for r in train]
    models = {
        'char_tfidf_logreg': make_pipeline(TfidfVectorizer(analyzer='char',ngram_range=(1,3),min_df=2,max_features=50000,sublinear_tf=True), LogisticRegression(C=1.0,max_iter=1000,solver='liblinear',random_state=20261023)),
        'char_tfidf_naive_bayes': make_pipeline(TfidfVectorizer(analyzer='char',ngram_range=(1,3),min_df=2,max_features=50000,sublinear_tf=True), MultinomialNB(alpha=1.0)),
        'length_only_logreg': make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=1000,solver='liblinear',random_state=20261023)),
    }
    report = {'scope':'public_recognition_source_holdout_only', 'human_rank_validation':False,
              'manifest_sha256':sha256(ROOT/'data/recognition.jsonl'),
              'selection':'Hyperparameters fixed before first run; validation not used for tuning; no human scores used.',
              'models':{}, 'majority_baseline':{}}
    predictions = []
    artifact_dir = ROOT/'models'
    artifact_dir.mkdir(exist_ok=True)
    def inputs(rs,name):
        return [[len(r['text'])] for r in rs] if name=='length_only_logreg' else [r['text'] for r in rs]
    with threadpool_limits(limits=4), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for name, model in models.items():
            model.fit(inputs(train,name),y)
            report['models'][name] = {}
            for split in ['validation','source_holdout']:
                subset = [r for r in rows if r['split']==split]
                probs = model.predict_proba(inputs(subset,name))[:,1]
                report['models'][name][split] = metrics([r['label'] for r in subset],probs)
                predictions.extend({'sample_id':r['sample_id'],'split':split,'model_id':name,'score':float(p),'label':r['label']} for r,p in zip(subset,probs))
            joblib.dump(model,artifact_dir/f'{name}.joblib')
    majority = collections.Counter(y).most_common(1)[0][0]
    for split in ['validation','source_holdout']:
        truth = [r['label'] for r in rows if r['split']==split]
        report['majority_baseline'][split] = metrics(truth,[float(majority)]*len(truth))
    write_json(ROOT/'reports/cpu_baselines.json',report)
    write_jsonl(ROOT/'data/source_predictions.jsonl',predictions)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
