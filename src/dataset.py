from collections import Counter, defaultdict
from copy import copy
from pathlib import Path
from typing import Optional

import pandas as pd
import toolz


class TagAssociations:
    """
    Parse tag co-occurrence information from AVClass run
    """

    def __init__(self, file: Path):
        self.tag_counter = {}
        self.pair_counter = defaultdict(dict)
        self.pair_norm_freq = defaultdict(dict)

        with open(file, "rt") as fp:
            fp.readline()
            for line in fp:
                t1, t2, nt1, nt2, both, *_ = line.split()
                # how many times t1, t2 appear
                self.tag_counter[t1] = int(nt1)
                self.tag_counter[t2] = int(nt2)

                # how many times t1 and t2 appear together
                self.pair_counter[t1][t2] = int(both)
                self.pair_counter[t2][t1] = int(both)

                # frequency of t2 relative to t1
                self.pair_norm_freq[t1][t2] = int(both) / int(nt1)
                assert self.pair_norm_freq[t1][t2] <= 1

                # frequency of t1 relative to t2
                self.pair_norm_freq[t2][t1] = int(both) / int(nt2)  # both / t2
                assert self.pair_norm_freq[t2][t1] <= 1


class TagAugmenter:
    """
    Resolve and augment tags from prev and curr AVClass info on EMBER data.

    Procedure
    - `(prev = None, curr = None)`   -> nothing to do
    - `(prev = None, curr = Valid)`  -> use `curr` and augment `FAM` & `CLASS` tags with co-occurrence info, respecting a threshold
    - `(prev = Valid, curr = None)`  -> find tags which co-occur with `FAM:prev` or `UNK:prev`, respecting a threshold
    - `(prev = Valid, curr = Valid)` -> if `curr` & `prev` agree, just resolve `curr` as above, otherwise resolve both separately and concat the results
    """

    def __init__(self, assoc: TagAssociations, thr_co_occur: float):
        self.assoc = assoc
        self.thr_co_occur = thr_co_occur

    def resolve_prev(self, prev: str) -> dict:
        """
        add `prev` to out with value None
            (can be the single info if no other is found from co-occurrence)
            None indicates that we don't have frequency info
        add tags y which co-occur with `prev` and have freq(prev & y) / freq(prev) >= thr
        """
        out = {}

        # at the very least, if there's no info from assoc, add `prev` tag as FAM
        out[f"FAM:{prev}"] = None

        for kind in ["UNK", "FAM"]:
            tag = f"{kind}:{prev}"
            out_links = filter_by_thr(self.assoc.pair_norm_freq[tag], thr=self.thr_co_occur)
            for new_tag, new_val in out_links.items():
                out[tag, new_tag] = new_val

        return out

    def resolve_curr(self, curr: dict) -> dict:
        """
        for x of kind FAM or CLASS:
            for pairs (x, y) with co-occurrence freq(x & y) / freq(x) >= thr:
                add to out (x, y): freq(x & y) / freq(x)
        """
        out = {}

        # for FAM & CLASS tags only
        # find other tags which co-occur >= thr (rel. to source tag)
        for tag_src, val_src in filter_by_kinds(curr, kinds=["FAM", "CLASS"]).items():
            out_links = filter_by_thr(self.assoc.pair_norm_freq[tag_src], thr=self.thr_co_occur)
            for tag_dst, val_dst in out_links.items():
                if tag_dst in curr:
                    continue
                out[tag_src, tag_dst] = val_dst

        return out

    def resolve_both(self, prev: str, curr: dict) -> dict:
        """
        if prev & curr agree, just resolve curr
        o/w, resolve both separately and concat the results
        """
        # prev & curr agree
        if f"FAM:{prev}" in curr or f"UNK:{prev}" in curr:
            return self.resolve_curr(curr)

        # prev & curr disagree
        r_prev = self.resolve_prev(prev)
        r_curr = self.resolve_curr(curr)
        assert set(r_prev.keys()) & set(r_curr.keys()) == set()

        return {**r_prev, **r_curr}

    def resolve_final_tags(self, row: pd.Series):
        """
        Resolve final tags for a single sample represented as DataFrame row.
        """
        prev = copy(row["avclass_prev"])
        curr = copy(row["avclass_curr"])

        pn, cn = pd.notna(prev), pd.notna(curr)

        # no info from prev or curr
        if pn == cn == False:
            return None

        # got info only from prev
        if pn and not cn:
            return self.resolve_prev(prev)

        # got info only from curr
        if not pn and cn:
            return self.resolve_curr(curr)

        # got info from both prev and curr
        return self.resolve_both(prev, curr)


def filter_by_kinds(xs: dict, kinds: list[str]) -> dict:
    return toolz.dicttoolz.keyfilter(lambda kind: any(kind.startswith(k) for k in kinds), xs)


def filter_by_thr(xs: dict, thr: float) -> dict:
    return toolz.dicttoolz.valfilter(lambda val: val >= thr, xs)


def normalise_wrt(kind: str, tag_scores: dict) -> Optional[dict]:
    """
    Normalise a (tag: score) mapping constraining tags to `kind`.
    """
    tmp = {k: v for k, v in tag_scores.items() if k.startswith(kind)}
    n = sum(tmp.values())
    if n == 0:
        return None
    return {k: v / n for k, v in tmp.items()}


def get_tag_ranking(
    tag_scores: dict, co_occurrence: dict, tag_kind: str, return_scores: bool = True
) -> Optional[list]:
    """
    Obtain a ranking of the `tag_kind` tags using co-occurrence information.
    If `return_scores` is True, then also return the [(tag, score)], o/w return just [tag].
    Currently supported kinds = {CLASS, FAM}
    """
    assert tag_kind in {"CLASS", "FAM"}
    acc = Counter()
    prior = Counter()

    # get normalised {tag -> score} w.r.t. vendor votes
    if fam := normalise_wrt("FAM:", tag_scores):
        prior.update(fam)
    if clz := normalise_wrt("CLASS:", tag_scores):
        prior.update(clz)

    for tag, score in co_occurrence.items():
        # singleton family, e.g. {"FAM:zusy": None}
        if score is None:
            continue

        src, dst = tag

        # src (fam) -> dst (`tag_kind`)
        # score is freq(dst|src)
        if src.startswith("FAM:") and dst.startswith(f"{tag_kind}:"):
            acc[dst] += prior[src] * score

    out = prior + acc
    out = Counter({k: v for k, v in out.items() if k.startswith(f"{tag_kind}:")}).most_common()

    if not return_scores:
        out = [x[0] for x in out]

    if len(out) == 0:
        return None

    return out


def get_most_prevalent_tag(tag_scores: dict, tag_kind: str, return_score: bool = True) -> list:
    """
    Get most prevalent tag (by scores and kind), without involving co-occurrence.

    Example:
        scores = {"FAM:a": 2, "FAM:b": 5, "CLASS:c": 3, "CLASS:d": 9}
        tag_kind = FAM   => [FAM:b]
        tag_kind = CLASS => [CLASS:d]
    """
    out = normalise_wrt(f"{tag_kind}:", tag_scores)
    if not out:
        return None

    tag, score = Counter(out).most_common()[0]

    if not return_score:
        return [tag]
    return [(tag, score)]
