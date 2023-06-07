import argparse
from functools import partial
from multiprocessing import Pool

import numpy as np
from numba import njit, prange

from leaf_similarity.leaf_pred_helper import load_leaf_dataset, read_metadata


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


def get_families_to_count(metadata):
    families = metadata.avclass.value_counts()
    return dict(families)


def cut_families_at_threshold(family_ty_count, threshold):
    return {k: v for k, v in family_ty_count.items() if v >= threshold}


def get_entry_family_by_index(metadata, index):
    return metadata.iloc[index].avclass


def get_match_count_percentage(
    percentage, metadata, families_to_count, len_train, similarities_vector, j
):
    index = len_train + j
    entry_class = get_entry_family_by_index(metadata, index)
    if entry_class not in families_to_count:
        return entry_class, -1
    target_percentage = int(percentage / 100 * families_to_count[entry_class])
    row = [(x[1], x[0]) for x in enumerate(similarities_vector)]
    top_hits = sorted(row, reverse=True)[:target_percentage]
    counter = 0
    for hit in top_hits:
        hit_class = get_entry_family_by_index(metadata, hit[1])
        if entry_class == hit_class:
            counter += 1
    return entry_class, counter / target_percentage * 100


def compute_families_percentage_similarities_statistics_test_vs_train_test(
    leaves_train_dataset_path,
    leaves_test_dataset_path,
    ember_metadata_path,
    output_path,
    target_percentage=10,
    minimum_count_threshold=100,
):
    leaves_train = load_leaf_dataset(leaves_train_dataset_path)
    leaves_test = load_leaf_dataset(leaves_test_dataset_path)
    all_leaves = np.concatenate((leaves_train, leaves_test), axis=0)

    metadata = read_metadata(ember_metadata_path)
    families_to_count = get_families_to_count(metadata)
    families_to_count = cut_families_at_threshold(families_to_count, minimum_count_threshold)

    results = {c: [] for c in families_to_count}

    len_train = len(leaves_train)
    start = 0
    end = 1000
    while end <= len(leaves_test):
        similarities = get_similarities_for_target_leaves(leaves_test, all_leaves, start, end)
        my_partial = partial(
            get_match_count_percentage, target_percentage, metadata, families_to_count, len_train
        )
        js = []
        simils = []
        for j in range(start, end):
            js.append(j)
            simils.append(similarities[j - start])
        with Pool() as pool:
            parallel_results = pool.starmap(
                my_partial, [(simils[x], js[x]) for x in range(len(js))]
            )
        parallel_results = sorted(parallel_results, key=lambda x: x[0])
        for r in parallel_results:
            if r[1] == -1:
                continue
            results[r[0]].append(r[1])
        start += 1000
        end += 1000
    with open(output_path, "w") as f:
        for c, counts in results.items():
            f.write(f"{c}, {np.mean(counts)}" + "\n")


def main(args):
    compute_families_percentage_similarities_statistics_test_vs_train_test(
        args.leaf_train_dataset_path,
        args.leaf_test_dataset_path,
        args.ember_metadata_path,
        args.output_path,
        args.target_percentage,
        args.minimum_count_threshold,
    )


if __name__ == "__main__":
    description = (
        "Compute the statistics of retrieval per class having the train and test leaf datasets"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        default="",
        help="Output path for the result file.",
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
        "--ember_metadata_path",
        type=str,
        required=False,
        default="",
        help="Path to the ember metadata.",
    )
    parser.add_argument(
        "--target_percentage",
        type=int,
        required=False,
        default=10,
        help="What percentage of the samples from a class should be the target number for a sample in that specific class.",
    )
    parser.add_argument(
        "--minimum_count_threshold",
        type=int,
        required=False,
        default=100,
        help="Ignore the class that has less than the threshould number of entries.",
    )
    args = parser.parse_args()
    main(args)
