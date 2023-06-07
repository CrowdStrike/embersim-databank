import argparse
import pickle
from multiprocessing import Pool

import numpy as np
from numba import njit, prange

from leaf_similarity.leaf_pred_helper import load_leaf_dataset


@njit(parallel=True)
def get_similarities_for_target_leaves(target_leaves, all_leaves, start, end):
    length_all_leaves = len(all_leaves)
    lenght_target_leaves = len(target_leaves)
    similarities = [np.zeros(length_all_leaves) for _ in range(end - start)]
    for i, leaves_one in enumerate(target_leaves):
        if i < start or i >= end:
            continue
        for j in prange(length_all_leaves):
            leaves_two = all_leaves[j]
            similarity = np.count_nonzero(leaves_one == leaves_two)
            similarities[i - start][j] = similarity
        if (i + 1) % 100 == 0:
            print(f"Done {i+1} entries from {lenght_target_leaves}..")
    return similarities


def get_top_similarities(i, similarities):
    row = [(x[1], x[0]) for x in enumerate(similarities)]
    return i, sorted(row, reverse=True)[:100]


def get_top_100_similar_test_vs_train_test(
    leaves_train_dataset_path, leaves_test_dataset_path, output_folder_path
):
    leaves_train = load_leaf_dataset(leaves_train_dataset_path)
    leaves_test = load_leaf_dataset(leaves_test_dataset_path)
    all_leaves = np.concatenate((leaves_train, leaves_test), axis=0)

    start = 0
    end = 1000
    while end <= len(leaves_test):
        similarities = get_similarities_for_target_leaves(leaves_test, all_leaves, start, end)
        with Pool() as pool:
            results = pool.starmap(
                get_top_similarities, [(i, x) for i, x in enumerate(similarities)]
            )
        results = sorted(results, key=lambda x: x[0])
        with open(f"{output_folder_path}/{start}.pkl", "wb") as g:
            pickle.dump(results, g)
        start += 1000
        end += 1000
        print(f"Done {start}..")


def get_top_100_similarities_unlabelled_vs_train_test(
    leaves_train_dataset_path,
    leaves_test_dataset_path,
    leaves_unlabelled_dataset_path,
    output_folder_path,
):
    leaves_train = load_leaf_dataset(leaves_train_dataset_path)
    leaves_test = load_leaf_dataset(leaves_test_dataset_path)
    leaves_unlabelled = load_leaf_dataset(leaves_unlabelled_dataset_path)
    all_leaves = np.concatenate((leaves_train, leaves_test), axis=0)

    start = 0
    end = 1000
    while end <= len(leaves_unlabelled):
        similarities = get_similarities_for_target_leaves(leaves_unlabelled, all_leaves, start, end)
        with Pool() as pool:
            results = pool.starmap(
                get_top_similarities, [(i, x) for i, x in enumerate(similarities)]
            )
        results = sorted(results, key=lambda x: x[0])
        with open(f"{output_folder_path}/{start}.pkl", "wb") as g:
            pickle.dump(results, g)
        start += 1000
        end += 1000
        print(f"Done {start}..")


def get_top_100_similarities_unlabelled_vs_train(
    leaves_train_dataset_path, leaves_unlabelled_dataset_path, output_folder_path
):
    leaves_train = load_leaf_dataset(leaves_train_dataset_path)
    leaves_unlabelled = load_leaf_dataset(leaves_unlabelled_dataset_path)

    start = 0
    end = 1000
    while end <= len(leaves_unlabelled):
        similarities = get_similarities_for_target_leaves(
            leaves_unlabelled, leaves_train, start, end
        )
        with Pool() as pool:
            results = pool.starmap(
                get_top_similarities, [(i, x) for i, x in enumerate(similarities)]
            )
        results = sorted(results, key=lambda x: x[0])
        with open(f"{output_folder_path}/{start}.pkl", "wb") as g:
            pickle.dump(results, g)
        start += 1000
        end += 1000
        print(f"Done {start}..")


def main(args):
    if args.command == "test_vs_train_test":
        get_top_100_similar_test_vs_train_test(
            args.leaf_train_dataset_path, args.leaf_test_dataset_path, args.output_folder_path
        )
    elif args.command == "unlabelled_vs_train_test":
        get_top_100_similarities_unlabelled_vs_train_test(
            args.leaf_train_dataset_path,
            args.leaf_test_dataset_path,
            args.leaf_unlabelled_dataset_path,
            args.output_folder_path,
        )
    elif args.command == "unlabelled_vs_train":
        get_top_100_similarities_unlabelled_vs_train_test(
            args.leaf_train_dataset_path, args.leaf_unlabelled_dataset_path, args.output_folder_path
        )
    else:
        print(
            "Command not supported. Choose from: test_vs_train_test; \
                unlabelled_vs_train_test; unlabelled_vs_train"
        )


if __name__ == "__main__":
    description = "Having two leaf predictions datasets A and B compute the top 100 similar entries from B for each sample in A."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        "--command",
        type=str,
        required=True,
        help="What datasets to compare; supported: test_vs_train_test; \
            unlabelled_vs_train_test; unlabelled_vs_train;",
    )
    parser.add_argument(
        "-o",
        "--output_folder_path",
        type=str,
        required=True,
        default="",
        help="Output path for the results; they will be saved in multiple pickle files.",
    )
    parser.add_argument(
        "--leaf_train_dataset_path",
        type=str,
        required=False,
        default="",
        help="Path to the leaf predictions train dataset.",
    )
    parser.add_argument(
        "--leaf_test_dataset_path",
        type=str,
        required=False,
        default="",
        help="Path to the leaf predictions test dataset.",
    )
    parser.add_argument(
        "--leaf_unlabelled_dataset_path",
        type=str,
        required=False,
        default="",
        help="Path to the leaf predictions unlabelled dataset.",
    )
    args = parser.parse_args()
    main(args)
