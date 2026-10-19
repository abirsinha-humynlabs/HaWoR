#!/usr/bin/env bash
# Apply pp_correct.py to an ALREADY uploaded long clip (run before the correction was added), regenerate
# its QC and re-upload the keypoints npz + qc/ to the same v3/ prefix, then verify S3 == local.
#   JOBS=<tsv> fix_uploaded.sh <clip>
set -eo pipefail
JOBS=${JOBS:?JOBS=<tsv> required}; C=${1:?clip}
IFS=$'\t' read -r _ SRC DST < <(grep -v '^#' "$JOBS" | awk -F'\t' -v c="$C" '$1==c')
[ -n "$DST" ] || { echo "no job for $C"; exit 1; }
B=stage-humyn-egocentric-stereo-data; P=labelling_results/hand_pose_HaWoR/v3/$DST/
O=~/hawor_runs/out/${C}_v3; V=/opt/dlami/nvme/fix/$C; N=$O/${C}_v3_hand21_keypoints.npz
mkdir -p "$V"
[ -s "$V/left_rectified.mp4" ] || AWS_PROFILE=prod aws s3 cp "$SRC/left_rectified.mp4" "$V/left_rectified.mp4" --only-show-errors
source ~/miniconda3/etc/profile.d/conda.sh; conda activate hawor
python ~/projects/HaWoR/delivery/long/pp_correct.py --npz "$N"
python ~/hawor_runs/scripts/qc_hawor.py --npz "$N" --video "$V/left_rectified.mp4" --clip "${C}_v3" --out "$O/qc" > /dev/null
python -c "import json,sys;q=json.load(open(sys.argv[1]));r=q['reproj']['median_px'];assert r<1,r;print('GATE reproj',r,'OK')" "$O/qc/${C}_v3_qc.json"
export AWS_PROFILE=stage
aws s3 cp "$N" "s3://$B/$P" --only-show-errors; aws s3 cp "$O/qc/" "s3://$B/${P}qc/" --recursive --only-show-errors
want=$(cd "$O" && { find . -maxdepth 1 -type f -name "${C}_v3_*" -printf '%P %s\n'; find qc -type f -printf '%p %s\n'; } | sort)
have=$(aws s3api list-objects-v2 --bucket $B --prefix "$P" --query 'Contents[].[Key,Size]' --output text | awk -F'\t' -v p="$P" '{k=$1; sub("^"p,"",k); print k, $2}' | sort)
[ "$want" = "$have" ] || { echo "VERIFY MISMATCH for $C"; diff <(echo "$want") <(echo "$have"); exit 1; }
rm -rf "$V"; echo "fixed + re-uploaded $C ($(echo "$have" | wc -l) files) -> s3://$B/$P"
