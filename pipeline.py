# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


def prepare_for_ohe_function(
        X, cols_to_keep, cols_to_replace_not_set
):
    X = X[cols_to_keep].copy()
    X = X.fillna('unknown')
    X = X.replace('', 'unknown')
    X[cols_to_replace_not_set] = (
        X[cols_to_replace_not_set]
        .replace('(not set)', 'unknown')
    )
    return X


def make_pipeline(columns):
    cols_to_keep = [
        col for col in columns
        if col.startswith(('utm_', 'device_', 'geo_'))
           and col not in ('device_model', 'device_screen_resolution')
    ]
    cols_to_replace_not_set = [
        col for col in cols_to_keep
        if not col.startswith('utm_')
    ]
    prepare_for_ohe = FunctionTransformer(
        prepare_for_ohe_function,
        kw_args={
            'cols_to_keep': cols_to_keep,
            'cols_to_replace_not_set': cols_to_replace_not_set
        },
        validate=False
    )

    ohe = OneHotEncoder(
        min_frequency=0.0001,
        handle_unknown='infrequent_if_exist',
        sparse_output=True
    )
    # lr = LogisticRegression(n_jobs=1)
    rf = RandomForestClassifier(
        max_depth=60,
        max_samples=0.5,
        min_samples_leaf=5,
        n_estimators=200,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ('prepare_for_ohe', prepare_for_ohe),
        ('ohe', ohe),
        ('model', rf)
    ])
    return pipeline
