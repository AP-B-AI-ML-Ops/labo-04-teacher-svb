import os
import pickle
import pandas as pd

from prefect import flow, task

from sklearn.feature_extraction import DictVectorizer

@task
def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

@task
def read_dataframe(filename: str):
    df = pd.read_csv(filename)

    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['pulocationid', 'dolocationid']
    df[categorical] = df[categorical].astype(str)

    return df

@task
def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    df['PU_DO'] = df['pulocationid'] + '_' + df['dolocationid']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv


@flow
def prep_flow(data_path: str, dest_path: str):
    # Load parquet files
    df_train = read_dataframe(
        os.path.join(data_path, "yellow-2021-01.csv")
    )
    df_val = read_dataframe(
        os.path.join(data_path, "yellow-2021-02.csv")
    )
    df_test = read_dataframe(
        os.path.join(data_path, "yellow-2021-03.csv")
    )

    # Extract the target
    target = 'tip_amount'
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    # Fit the DictVectorizer and preprocess data
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))
