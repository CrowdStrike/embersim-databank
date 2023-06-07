import glob
import pickle

import ember
import xgboost as xgb


def load_dataset(data_dir):
    X_train, y_train, X_test, y_test = ember.read_vectorized_features(data_dir)
    train_rows = y_train != -1
    unlabelled_rows = y_train == -1
    return (
        X_train[train_rows],
        y_train[train_rows],
        X_test,
        y_test,
        X_train[unlabelled_rows],
        y_train[unlabelled_rows],
    )


def load_model(path):
    model = xgb.Booster()
    model.load_model(path)
    return model


def load_leaf_dataset(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_tops_dataset(tops_dataset_path):
    tops = []
    for file in glob.glob(f"{tops_dataset_path}/*.pkl"):
        with open(file, "rb") as f:
            tops.extend([x[1] for x in pickle.load(f)])
    return tops


def read_metadata(metadata_path):
    metadata = ember.read_metadata(metadata_path)
    metadata["avclass"] = metadata["avclass"].fillna("clean")
    return metadata[metadata.label != -1].reset_index(drop=True)
