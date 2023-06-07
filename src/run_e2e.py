import subprocess
from itertools import product
from pathlib import Path

from loguru import logger

HERE = Path(__file__).parent

DATA = HERE / "../data/processed"
assert DATA.exists()

RANK_BY = "FAM"

OUT = DATA / "eval-results" / f"{RANK_BY.lower()}-ranking"
assert OUT.exists()

SCRIPT = HERE / "e2e.py"
assert SCRIPT.exists()

BASE_CMD = {
    "--input-dataframe": str((DATA / "ember_with_avclass_dataset.pkl").resolve()),
    "--assoc-file": str((DATA / "ember_vt_detections.alias").resolve()),
    "--sim-results": str((DATA / "xgb-sim-results/test_vs_train_test.pkl").resolve()),
    "--rank-by": RANK_BY,
    "--topk": 100,
}

# "grid search" params
THRS_CO_OCCUR = [0.1, 0.5, 0.75, 0.9]
REL_TYPES = ["exact", "edit", "iou"]


def dict_to_cmd(xs):
    return ["python3", str(SCRIPT.resolve())] + list(map(str, sum(xs.items(), ())))


CMDS = []

for thr_co_occur, rel_type in product(THRS_CO_OCCUR, REL_TYPES):
    CMDS.append(
        dict_to_cmd(
            {
                **BASE_CMD,
                "--thr-co-occur": thr_co_occur,
                "--relevance-type": rel_type,
                "--output-results-file": (
                    OUT
                    / f"test_vs_traintest_prec_at_100_rank_{RANK_BY}_occur_{thr_co_occur}_rel_{rel_type}.pkl"
                ),
            }
        )
    )

logger.info(f"Got {len(CMDS)} configurations")

for i, cmd in enumerate(CMDS, start=1):
    logger.info(f"[{i}/{len(CMDS)}] Starting {cmd=}")
    p = subprocess.Popen(cmd)
    p.wait()
