import argparse
import pickle

import xgboost as xgb

from leaf_similarity.leaf_pred_helper import load_dataset, load_model


def compute_leaf_predictions(X, model, output_path):
    data = xgb.DMatrix(X)
    predicts = model.predict(data, pred_leaf=True)
    with open(output_path, "wb") as g:
        pickle.dump(predicts, g)


def compute_predictions(X, model, output_path):
    data = xgb.DMatrix(X)
    predicts = model.predict(data)
    with open(output_path, "wb") as g:
        pickle.dump(predicts, g)


def main(args):
    if args.command == "compute_leaf_predictions_train":
        X_train, _, _, _, _, _ = load_dataset(args.dataset_path)
        model = load_model(args.model_path)
        compute_leaf_predictions(X_train, model, args.output_path)
    elif args.command == "compute_leaf_predictions_test":
        _, _, X_test, _, _, _ = load_dataset(args.dataset_path)
        model = load_model(args.model_path)
        compute_leaf_predictions(X_test, model, args.output_path)
    elif args.command == "compute_leaf_predictions_unlabelled":
        _, _, _, _, X_unlabelled, _ = load_dataset(args.dataset_path)
        model = load_model(args.model_path)
        compute_leaf_predictions(X_unlabelled, model, args.output_path)
    elif args.command == "compute_predictions_unlabelled":
        _, _, _, _, X_unlabelled, _ = load_dataset(args.dataset_path)
        model = load_model(args.model_path)
        compute_predictions(X_unlabelled, model, args.output_path)
    else:
        print(
            "Command not supported. Choose from: compute_leaf_predictions_train, \
                    compute_leaf_predictions_test, compute_leaf_predictions_unlabelled, and compute_predictions_unlabelled"
        )


if __name__ == "__main__":
    description = "Compute leaf predictions or normal predictions using an XGBoost model."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-c", "--command", type=str, required=True, help="What task do you want to perform"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        default="",
        help="Output path for the result",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="XGBoost model path. Should be a model represented as JSON.",
    )
    parser.add_argument("-e", "--dataset_path", type=str, required=True, help="Ember dataset path")
    args = parser.parse_args()
    main(args)
