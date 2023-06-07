"""
Run end-to-end evaluation pipeline.
"""
import argparse
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

tqdm.pandas()

from dataset import TagAssociations, TagAugmenter, get_most_prevalent_tag, get_tag_ranking
from evaluation import RelevanceAtK, RelevanceMethods

LOADER_BY_EXT = {".csv": pd.read_csv, ".pkl": pd.read_pickle}


def load_dataset(args):
    logger.info(f"Loading main dataset (EMBER + AVClass) from {args.input_dataframe}")
    df = LOADER_BY_EXT[args.input_dataframe.suffix](args.input_dataframe)
    logger.info(f"Loaded. {df.shape=}")

    aug_df = df  # df.copy()

    if args.rank_top_only:
        logger.info(f"Getting most prevalent {args.rank_by} tag (w/o co-occurrence)")
        aug_df["TAG_RANKS"] = aug_df.query("avclass_curr.notna()").progress_apply(
            lambda row: get_most_prevalent_tag(
                tag_scores=row["avclass_curr"], tag_kind=args.rank_by, return_score=False
            ),
            axis=1,
        )
        return aug_df

    logger.info(f"Constructing tag co-occurrence info from {args.assoc_file}")
    tag_assoc = TagAssociations(args.assoc_file)

    logger.info("Augmenting dataset given AVClass tag co-occurrence info")
    logger.info(f"Co-occurrence threshold: {args.thr_co_occur}")
    tag_aug = TagAugmenter(tag_assoc, thr_co_occur=args.thr_co_occur)
    aug_df["EXTRA"] = aug_df.progress_apply(tag_aug.resolve_final_tags, axis=1)

    # in order to obtain ground truths (tag rankings),
    # we must have both avclass tags and extra tag co-occurrence info
    logger.info(f"Constructing {args.rank_by} tag ranks")
    aug_df["TAG_RANKS"] = aug_df.query("avclass_curr.notna() & EXTRA.notna()").progress_apply(
        lambda row: get_tag_ranking(
            tag_scores=row["avclass_curr"],
            co_occurrence=row["EXTRA"],
            tag_kind=args.rank_by,
            return_scores=False,
        ),
        axis=1,
    )
    logger.info(f"Resulting dataset size: {len(aug_df)}")

    return aug_df


def load_sim_results(args):
    logger.info(f"Loading sim results from {args.sim_results}")
    sim_res_df = LOADER_BY_EXT[args.sim_results.suffix](args.sim_results)
    logger.info(f"Loaded sim results: {sim_res_df.shape}")

    return sim_res_df


def main(args):
    if args.relevance_type == "iou":
        relevance_func = RelevanceMethods.iou
    elif args.relevance_type == "exact":
        relevance_func = RelevanceMethods.exact_match
    elif args.relevance_type == "edit":
        relevance_func = RelevanceMethods.edit_distance
    else:
        raise ValueError(f"Unknown {args.relevance_type=}")

    aug_df = load_dataset(args)
    sim_res_df = load_sim_results(args)

    logger.info(f"Running evaluation (relevance: {args.relevance_type})")
    # results will be a dataframe with `sha256, result`
    results = RelevanceAtK(
        aug_df,
        sim_res_df,
        k=args.topk,
        relevance_func=relevance_func,
    ).run()

    logger.info(f"Dumping results to {args.output_results_file}")
    results.to_pickle(args.output_results_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dataframe",
        type=Path,
        required=True,
        help="Path to the input EMBER dataframe with AVClass columns",
    )
    parser.add_argument(
        "--assoc-file",
        type=Path,
        required=False,
        help="Path to the AVClass alias file containing tag associations (co-occurrences)",
    )
    parser.add_argument(
        "--sim-results",
        type=Path,
        required=True,
        help="Path to a dataframe containing query and top-N similarity search results",
    )
    parser.add_argument(
        "--thr-co-occur",
        type=float,
        required=False,
        help="Tag co-occurrence threshold when enriching tag information from AVClass",
    )
    parser.add_argument(
        "--rank-by",
        type=str,
        choices=["FAM", "CLASS"],
        required=True,
        help="Tag kind to rank when preparing evaluation",
    )
    parser.add_argument(
        "--rank-top-only",
        action="store_true",
        help="If true, only keep the most prevalent tag (for evaluation)",
    )
    parser.add_argument(
        "--relevance-type",
        type=str,
        choices=["exact", "iou", "edit"],
        required=True,
        help="Function to compute relevance between query and hit tag items.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        required=False,
        default=100,
        help="Compute relevance for this top-K results",
    )
    parser.add_argument(
        "--output-results-file",
        type=Path,
        required=True,
        help="Path to the output dataframe with results",
    )

    args = parser.parse_args()

    main(args)
