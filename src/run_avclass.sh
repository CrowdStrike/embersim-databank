#!/usr/bin/env bash

AVCLASS_DIR='../../../avclass.git'

python3 $AVCLASS_DIR/avclass/labeler.py \
    -f ./ember_vt_detections.jsonl \
    -o ./ember_avclass_results.txt \
    -hash sha256 \
    -aliasdetect \
    -avtags \
    -t \
    -vtt \
    -stats

# -aliasdetect  produce aliases file at end
# -avtags       extracts tags per av vendor
# -t            output all tags, not only the family.
# -vtt          include VT tags in the output
# -stats        produce 1 file with stats per category (File, Class, Behavior, Family, Unclassified)