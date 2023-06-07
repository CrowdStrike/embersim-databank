import pandas as pd
import textdistance
from tqdm import tqdm


def _early_validate(func):
    """
    Check if either of class lists are null, in which case we can exit early and provide the result.

    If both class lists are null, we consider the sample to be clean, so the relevance is 100%.
    If only one class list is null, we consider a total mismatch, so the relevance is 0%.
    If both class lists are valid, use the wrapped `func`, which implements a proper scoring method.
    """

    def inner(query_classes: list[str], other_classes: list[str], **kwargs):
        q_ok = isinstance(query_classes, list) and pd.notna(query_classes).all()
        o_ok = isinstance(other_classes, list) and pd.notna(other_classes).all()

        # if we don't get class info for neither source / other,
        # then source & other are most likely clean,
        # so we consider the result valid and return 1
        if q_ok == o_ok == False:
            return (q_ok, o_ok, 1.0)

        # if either one the class ranks are valid, but the other is not,
        # then we consider a label mismatch, so we return 0
        if q_ok != o_ok:
            return (q_ok, o_ok, 0.0)

        # at this point, both class lists are valid, employ wrapped `func`
        return (q_ok, o_ok, func(query_classes, other_classes, **kwargs))

    return inner


class RelevanceMethods:
    @staticmethod
    @_early_validate
    def exact_match(query_classes: list[str], other_classes: list[str]) -> float:
        """
        Exact match between class lists.
        """
        return int(query_classes == other_classes)

    @staticmethod
    @_early_validate
    def iou(query_classes: list[str], other_classes: list[str]) -> float:
        """
        Intersection Over Union between class lists.
        """
        q = set(query_classes)
        o = set(other_classes)

        return len(q & o) / len(q | o)

    @staticmethod
    @_early_validate
    def edit_distance(query_classes: list[str], other_classes: list[str]) -> float:
        """
        Normalised Edit Distance between class lists.
        """
        func = textdistance.damerau_levenshtein
        dist = func.distance(query_classes, other_classes)
        norm = func.maximum(query_classes, other_classes)

        return 1 - dist / norm


class RelevanceAtK:
    def __init__(
        self,
        aug_df: pd.DataFrame,
        df_to_eval: pd.DataFrame,
        k: int,
        relevance_func: callable,
    ):
        self.aug_df = aug_df.set_index("sha256")
        self.df_to_eval = df_to_eval
        self.k = k
        self.relevance_func = relevance_func

    def run(self) -> pd.DataFrame:
        """
        Run the evaluation and return a DataFrame with `sha256,results` as columns.
        `results` is a list of tuples with info for each sample, as given by the relevance function.
        """
        out = []

        for _, row in tqdm(
            self.df_to_eval.iterrows(),
            total=len(self.df_to_eval),
            desc="Eval",
        ):
            needle = row["needle_sha256"]
            hits = row["hits_sha256"]
            true = self.aug_df.loc[needle, "TAG_RANKS"]
            pred = self.aug_df.loc[hits, "TAG_RANKS"].to_list()

            out.append(
                {
                    "sha256": needle,
                    "results": [self.relevance_func(true, p) for p in pred[: self.k]],
                }
            )

        return pd.DataFrame(out)
