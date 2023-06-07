import argparse

import pandas as pd

from leaf_similarity.leaf_pred_helper import load_dataset, load_tops_dataset


def get_similar_shas_test_vs_train_test(
    tops_folder_path, ember_metadata_path, output_path, with_similarity_scores
):
    tops = load_tops_dataset(tops_folder_path)

    df = pd.read_csv(ember_metadata_path)
    df = df.drop(df[df["label"] == -1].index).reset_index(drop=True)

    _, y_train, _, _, _, _ = load_dataset()
    len_train = len(y_train)

    result_dict = {"needle_sha256": [], "hits_sha256": []}
    for i, top in enumerate(tops):
        result_dict["needle_sha256"].append(df.iloc[len_train + i]["sha256"])
        if with_similarity_scores:
            result_dict["hits_sha256"].append([(df.iloc[x[1]]["sha256"], x[0]) for x in top])
        else:
            result_dict["hits_sha256"].append([df.iloc[x[1]]["sha256"] for x in top])
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(output_path)


def get_similar_shas_unlabelled_vs_train_test(
    tops_folder_path, ember_metadata_path, output_path, with_similarity_scores
):
    tops = load_tops_dataset(tops_folder_path)

    df = pd.read_csv(ember_metadata_path)
    df_train_test = df.drop(df[df["label"] == -1].index).reset_index(drop=True)
    df_unlabelled = df[df["label"] == -1].reset_index(drop=True)

    result_dict = {"needle_sha256": [], "hits_sha256": []}
    for i, top in enumerate(tops):
        result_dict["needle_sha256"].append(df_unlabelled.iloc[i]["sha256"])
        if with_similarity_scores:
            result_dict["hits_sha256"].append(
                [(df_train_test.iloc[x[1]]["sha256"], x[0]) for x in top]
            )
        else:
            result_dict["hits_sha256"].append([df_train_test.iloc[x[1]]["sha256"] for x in top])
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(output_path)


def get_similar_shas_unlabelled_vs_train(
    tops_folder_path, ember_metadata_path, output_path, with_similarity_scores
):
    tops = load_tops_dataset(tops_folder_path)

    df = pd.read_csv(ember_metadata_path)
    df_train_test = df.drop(df[df["label"] == -1].index).reset_index(drop=True)
    df_unlabelled = df[df["label"] == -1].reset_index(drop=True)

    result_dict = {"needle_sha256": [], "hits_sha256": []}
    for i, top in enumerate(tops):
        result_dict["needle_sha256"].append(df_unlabelled.iloc[i]["sha256"])
        if with_similarity_scores:
            result_dict["hits_sha256"].append(
                [(df_train_test.iloc[x[1]]["sha256"], x[0]) for x in top]
            )
        else:
            result_dict["hits_sha256"].append([df_train_test.iloc[x[1]]["sha256"] for x in top])
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(output_path)


def main(args):
    if args.command == "test_vs_train_test":
        get_similar_shas_test_vs_train_test(
            args.tops_folder_path,
            args.ember_metadata_path,
            args.output_path,
            args.with_similarity_scores,
        )
    elif args.command == "unlabelled_vs_train_test":
        get_similar_shas_unlabelled_vs_train_test(
            args.tops_folder_path,
            args.ember_metadata_path,
            args.output_path,
            args.with_similarity_scores,
        )
    elif args.command == "unlabelled_vs_train":
        get_similar_shas_unlabelled_vs_train(
            args.tops_folder_path,
            args.ember_metadata_path,
            args.output_path,
            args.with_similarity_scores,
        )
    else:
        print(
            "Command not supported. Choose from: test_vs_train_test; \
                unlabelled_vs_train_test; unlabelled_vs_train"
        )


if __name__ == "__main__":
    description = "Having the top 100 similar dataset convert it into a csv with the SHA256 of the files. \
        You have the option to include the similarities scores as well."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        "--command",
        type=str,
        required=True,
        help="What dataset sources to be considered; supported: test_vs_train_test; \
            unlabelled_vs_train_test; unlabelled_vs_train;",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        default="",
        help="Output path for the results; this is a csv file path.",
    )
    parser.add_argument(
        "--tops_folder_path",
        type=str,
        required=False,
        default="",
        help="Path to the tops dataset.",
    )
    parser.add_argument(
        "--ember_metadata_path",
        type=str,
        required=False,
        help="Path to the ember metadata.",
    )
    parser.add_argument(
        "--with_similarity_scores",
        type=bool,
        required=False,
        default=False,
        help="To either output the similarity scores or not. Default false.",
    )
    args = parser.parse_args()
    main(args)
