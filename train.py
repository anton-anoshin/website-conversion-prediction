import datetime
import time

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

import pipeline
from pathlib import Path

def read_data(file):
    df = pd.read_pickle(file)
    X = df.drop(columns='target')
    y = df['target']
    return X, y


def validate(pipeline, X, y, cv):
    start = time.time()
    scores = cross_val_score(
        pipeline, X, y, cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2
    )
    end = time.time()
    # print(f'Модель: {type(pipeline.steps[-1][1]).__name__}')
    print(f'\nModel: {type(pipeline.named_steps["model"]).__name__}')
    print(f'Cross-validation duration: {(end - start) / 60:.1f} min')
    score_mean = scores.mean()
    print(f'ROC-AUC mean: {score_mean:.2f}')
    print(f'ROC-AUC std:  {scores.std():.4f}')
    return score_mean


def main():
    print('Session Target Action Prediction Pipeline')
    project_dir = Path(__file__).resolve().parent
    X, y = read_data(project_dir / 'data' / 'processed' / 'sessions_target.pkl')
    pipe = pipeline.make_pipeline(X.columns)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    print('\nCross-validation started.')
    roc_auc = validate(pipe, X, y, skf)
    print('\nFull-dataset fit started.')
    start = time.time()
    pipe.fit(X, y)
    print(f'Fit completed.\nFit duration: {(time.time() - start) / 60:.1f} min')
    print(f'\nSaving bundle to models/predict_target_pipeline.pkl.')
    joblib.dump({
        'pipeline': pipe,
        'metadata': {
            'name': 'Session Target Action Prediction Pipeline',
            'author': 'Anton Anoshin',
            'version': '1.0',
            'date': datetime.datetime.now(),
            'type': type(pipe.named_steps['model']).__name__,
            'roc_auc': round(roc_auc, 2),
            'threshold': 0.06,
            'threshold_method': 'max F1 on OOF train proba'
        }
    }, project_dir / 'models' / 'predict_target_pipeline.pkl')


if __name__ == '__main__':
    main()
