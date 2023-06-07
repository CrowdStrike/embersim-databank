"""
Parse AVClass results and augment and existing EMBER dataframe.
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_tag(tag: str) -> tuple[str, str]:
    """Example
    from FILE:os:windows|17
      to (FILE:os:windows, 17)
    """
    try:
        kind, votes = tag.split("|")
        return kind, int(votes)
    except ValueError as e:
        return None


def normalise_tags(tags: list) -> dict:
    assert isinstance(tags, list)
    if not tags:
        return None

    out = {}

    for kind, votes in tags:
        out[kind] = votes

    return out


def parse_avclass_results(results_file: Path) -> pd.DataFrame:
    avclass_results = []

    with open(results_file, "rt") as fp:
        for line in fp:
            sha256, vt_detections, *tags = line.strip().split("\t")

            if tags:
                assert len(tags) == 1
                tags = tags[0].split(",")
                tags = [parse_tag(t) for t in tags]
                tags = normalise_tags([t for t in tags if t is not None])
            else:
                tags = None

            assert isinstance(tags, dict) or tags is None

            try:
                vt_detections = int(vt_detections)
            except ValueError:
                vt_detections = None

            avclass_results.append({"sha256": sha256, "vt_detections": vt_detections, "tags": tags})

    return pd.DataFrame(avclass_results).astype(
        {"sha256": "string[pyarrow]", "vt_detections": "int32[pyarrow]", "tags": "object"}
    )


def main(args):
    ext = args.output_dataframe_path.suffix
    if ext not in {".csv", ".pickle"}:
        raise ValueError("Only csv or pickle extensions are supported")

    ember_df = pd.read_csv(args.ember_dataframe_csv, index_col=0)
    avclass_results_df = parse_avclass_results(args.avclass_results_file)

    # augmented dataframe: prev ember | curr avclass results
    renamer = {
        "avclass": "avclass_prev",  # previous avclass results
        "tags": "avclass_curr",  # current avclass results
    }
    aug_df = ember_df.merge(avclass_results_df, how="left", on="sha256").rename(renamer, axis=1)

    if ext == ".csv":
        aug_df.to_csv(args.output_dataframe_path, index=False)
    elif ext == ".pickle":
        aug_df.to_pickle(args.output_dataframe_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--avclass-results-file",
        type=Path,
        required=True,
        help="Path to the avclass results file (txt)",
    )
    parser.add_argument(
        "--ember-dataframe-csv",
        type=Path,
        required=True,
        help="Path to the original EMBER dataframe to be augmented with AVClass results",
    )
    parser.add_argument(
        "--output-dataframe-path",
        type=Path,
        required=True,
        help="Path to dump the output augmented dataframe to",
    )
    args = parser.parse_args()

    main(args)
