import argparse

import numpy as np

from leaf_similarity.leaf_pred_helper import load_dataset, load_tops_dataset


def compute_statistics_for_binary_classif(
    tops_dataset_path, ember_dataset_dir, output_path, only_test=False
):
    tops = load_tops_dataset(tops_dataset_path)
    ks = [10, 50, 100]
    _, y_train, _, y_test, _, _ = load_dataset(ember_dataset_dir)
    y = np.concatenate((y_train, y_test), axis=0)
    stats = {k: [] for k in ks}
    for i, top in enumerate(tops):
        if only_test and i < len(y_train):
            continue
        needle_value = y[i]
        for k in ks:
            count = 0
            for element in top[:k]:
                if y[element[1]] == needle_value:
                    count += 1
            stats[k].append(count)
        if (i + 1) % 1000 == 0:
            print(f"Done {i+1}..")

    with open(output_path, "w") as f:
        f.write(f"K,Mean,Std" + "\n")
        f.write(f"10,{np.mean(stats[10])},{np.std(stats[10])}" + "\n")
        f.write(f"50,{np.mean(stats[50])},{np.std(stats[50])}" + "\n")
        f.write(f"100,{np.mean(stats[100])},{np.std(stats[100])}" + "\n")


def compute_statistics_for_binary_classif_per_class(
    tops_dataset_path, ember_dataset_dir, output_path, only_test=False
):
    tops = load_tops_dataset(tops_dataset_path)
    ks = [10, 50, 100]
    _, y_train, _, y_test, _, _ = load_dataset(ember_dataset_dir)
    y = np.concatenate((y_train, y_test), axis=0)
    stats = {0: {k: [] for k in ks}, 1: {k: [] for k in ks}}
    for i, top in enumerate(tops):
        if only_test and i < len(y_train):
            continue
        needle_value = y[i]
        for k in ks:
            count = 0
            for element in top[:k]:
                if y[element[1]] == needle_value:
                    count += 1
            stats[needle_value][k].append(count)
        if (i + 1) % 1000 == 0:
            print(f"Done {i+1}..")

    with open(output_path, "w") as f:
        for c, s in stats.items():
            f.write(f"Class,K,Mean,Std" + "\n")
            f.write(f"{c},10,{np.mean(s[10])},{np.std(s[10])}" + "\n")
            f.write(f"{c},50,{np.mean(s[50])},{np.std(s[50])}" + "\n")
            f.write(f"{c}100,{np.mean(s[100])},{np.std(s[100])}" + "\n")


def main(args):
    if args.command == "general":
        compute_statistics_for_binary_classif(
            args.tops_folder_path, args.ember_metadata_path, args.output_path, args.only_test
        )
    elif args.command == "per_class":
        compute_statistics_for_binary_classif_per_class(
            args.tops_folder_path, args.ember_metadata_path, args.output_path, args.only_test
        )
    else:
        print("Command not supported. Choose from: general, per_class")


if __name__ == "__main__":
    description = "Having the top 100 similar dataset compute the statistics for how many hits from top 100 have the same class."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        "--command",
        type=str,
        required=True,
        help="What type of stat do you want; supported: general, or per_class",
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
        "--only_test",
        type=bool,
        required=False,
        default=False,
        help="To either output the results just for the test subset or not. Default false.",
    )
    args = parser.parse_args()
    main(args)
