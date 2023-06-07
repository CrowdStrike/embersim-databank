import argparse
import pickle

import ember
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV


def load_dataset(data_dir):
    X_train, y_train, X_test, y_test = ember.read_vectorized_features(data_dir)
    train_rows = y_train != -1
    return X_train[train_rows], y_train[train_rows], X_test, y_test


def grid_search(X_train, y_train, results_output_path=""):
    classifier = xgboost.XGBClassifier(seed=42, tree_method="auto")
    params = {
        "max_depth": [6, 10, 12, 15, 17],
        "eta": [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
        "n_estimators": [1024, 1500, 2048],
        "colsample_bytree": [1, 0.7, 0.3],
    }

    grid_clf = GridSearchCV(estimator=classifier, param_grid=params, scoring="roc_auc", verbose=4)
    grid_clf.fit(X_train, y_train)

    if results_output_path:
        with open(results_output_path, "wb") as g:
            pickle.dump(grid_clf.cv_results_, g)

    print("Best parameters:", grid_clf.best_params_)
    print("Best score:", grid_clf.best_score_)


def train_xgb_model(
    max_depth, eta, n_estimators, colsample_bytree, dataset_path, model_output_path=""
):
    X_train, y_train, X_test, y_test = load_dataset(dataset_path)
    classifier = xgboost.XGBClassifier(
        seed=42,
        tree_method="auto",
        learning_rate=eta,
        max_depth=max_depth,
        n_estimators=n_estimators,
        colsample_bytree=colsample_bytree,
    )
    classifier.fit(X_train, y_train)
    if model_output_path:
        classifier.save_model(model_output_path)

    y_pred = classifier.predict(X_test)
    roc_score = roc_auc_score(y_test, y_pred)
    print(f"ROC_AUC score: {roc_score}")


def main(args):
    if args.command == "grid_search":
        X_train, y_train, _, _ = load_dataset(args.dataset_path)
        grid_search(X_train, y_train, args.output_path)
    elif args.command == "train_best_params":
        train_xgb_model(
            max_depth=17,
            eta=0.15,
            n_estimators=2048,
            colsample_bytree=1,
            model_output_path=args.output_path,
        )
    elif args.command == "train_custom_params":
        if (
            args.get("max_depth", -1) == -1
            or args.get("eta", -1) == -1
            or args.get("n_estimators", -1) == -1
            or args.get("colsample_bytree", -1) == -1
        ):
            print(
                "Please specify all the parameters as arguments: max_depth, eta, n_estimators, and colsample_bytree"
            )
            exit()
        train_xgb_model(
            max_depth=args.max_depth,
            eta=args.eta,
            n_estimators=args.n_estimators,
            colsample_bytree=args.colsample_bytree,
            model_output_path=args.output_path,
        )
    else:
        print(
            "Command not supported. Choose from: grid_search, train_best_params, train_custom_params"
        )


if __name__ == "__main__":
    description = "Perform a parameters grid search or train an xgboost model on Ember data"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        "--command",
        type=str,
        required=True,
        help="What task do you want to perform. It can be either grid_search, test_best_params, or train_custom_params",
    )
    parser.add_argument("-i", "--dataset_path", type=str, required=True, help="Ember dataset path")
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=False,
        default="",
        help="Output path for the result",
    )
    parser.add_argument(
        "-d", "--max_depth", type=int, required=False, help="max_depth argument for XGBoost"
    )
    parser.add_argument("-e", "--eta", type=float, required=False, help="eta argument for XGBoost")
    parser.add_argument(
        "-n", "--n_estimators", type=int, required=False, help="n_estimators argument for XGBoost"
    )
    parser.add_argument(
        "-s",
        "--colsample_bytree",
        type=float,
        required=False,
        help="colsample_bytree argument for XGBoost",
    )
    args = parser.parse_args()
    main(args)
