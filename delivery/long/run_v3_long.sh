#!/usr/bin/env bash
# v3 keypoints for one LONG clip (minutes, not ~60 s), run as overlapping ~60 s windows.
#
# Same model path and post-processing as delivery/run_v3_clip.sh stages 1-5, except:
#   * frames are extracted ONCE with HaWoR's own ffmpeg command, and each window is a folder of
#     symlinks to them, so HaWoR runs on exactly the frames it would extract itself;
#   * each window runs run_hawor_clip.py unchanged (detect -> motion -> SLAM -> infiller);
#     windows are WIN frames long with MARGIN frames of overlap context each side, and only the
#     core of each window is kept (merge_windows.py), so every frame comes from one window;
#   * post-processing, MediaPipe and the handedness vote run once on the merged clip;
#   * no head pose (none exists for these sources), no render (later), no S3 access;
#   * pp_correct.py moves kp3d/kp2d onto the true principal point (HaWoR assumes the image centre).
#
# Usage: run_v3_long.sh <clip> <src_dir>
#   <src_dir> holds left_rectified.mp4 + calibration.json (pre-fetched by long_batch.sh).
# Output: ~/hawor_runs/out/<clip>_v3/  ({clip}_v3_hand21_keypoints.npz, _3d_keypoints.npz, reports, qc/)
# Resumable: finished windows are skipped. WIN/MARGIN can be overridden from the environment.
set -eo pipefail
BASE="${1:?usage: run_v3_long.sh <clip> <src_dir>}"; SRC="${2:?src_dir}"
WIN=${WIN:-1800}; MARGIN=${MARGIN:-30}
C=${BASE}_v3
WK=~/hawor_runs/work/$C; FR=$WK/frames; OUT=~/hawor_runs/out/$C; SCRATCH=~/hawor_runs/scratch/$C
D=~/projects/HaWoR/delivery
VID=$SRC/left_rectified.mp4; CAL=$SRC/calibration.json
mkdir -p "$WK" "$OUT/qc" "$SCRATCH"
source ~/miniconda3/etc/profile.d/conda.sh
say(){ echo "=========== [$C] $(date -u +%H:%M:%S) $* ==========="; }
[ -s "$VID" ] && [ -s "$CAL" ] || { echo "FATAL: missing $VID or $CAL" >&2; exit 1; }
conda activate hawor
export CUDA_HOME=$CONDA_PREFIX PATH=$CONDA_PREFIX/bin:$PATH HAWOR_TRACK_FIX=1

say "0/5 frames + windows"
NV=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$VID")
if [ ! -f "$FR/.complete" ]; then
  rm -rf "$FR"; mkdir -p "$FR"
  # HaWoR's extract_frames() command (detect_track_video.py), only with a wider counter
  ffmpeg -loglevel error -i "$VID" -vf fps=30 -start_number 0 "$FR/%06d.jpg"
  touch "$FR/.complete"
fi
NF=$(find "$FR" -name '*.jpg' | wc -l)
[ "$NF" = "$NV" ] || { echo "FATAL: extracted $NF frames but the video has $NV - frame indices would not line up" >&2; exit 1; }
python - "$NF" "$WIN" "$MARGIN" "$VID" "$WK/windows.json" <<'PY'
import json, math, sys, cv2
n, win, m, vid, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
cap = cv2.VideoCapture(vid); W, H, fps = int(cap.get(3)), int(cap.get(4)), float(cap.get(5)); cap.release()
k = math.ceil(n / (win - 2 * m))                       # cores of <= win-2m frames, split evenly
b = [round(i * n / k) for i in range(k + 1)]
ws = [dict(name=f'w{i:03d}', core_start=b[i], core_end=b[i + 1],
           start=max(0, b[i] - m), end=min(n, b[i + 1] + m)) for i in range(k)]
json.dump(dict(frames=n, width=W, height=H, fps=fps, window_len=win, margin=m, windows=ws), open(out, 'w'), indent=1)
print(f'  {n} frames {W}x{H}@{fps:g} -> {k} windows of <= {max(w["end"] - w["start"] for w in ws)} frames')
PY

say "1/5 HaWoR inference per window (HAWOR_TRACK_FIX=1)"
python -c "import json,sys;[print(w['name'],w['start'],w['end']) for w in json.load(open(sys.argv[1]))['windows']]" \
  "$WK/windows.json" | while read -r NAME S E; do
  if [ -s "$WK/$NAME.out/${NAME}_raw_hand21_keypoints.npz" ]; then echo "  $NAME cached"; continue; fi
  rm -rf "${WK:?}/$NAME" "$WK/$NAME.out"; mkdir -p "$WK/$NAME/extracted_images"
  python - "$FR" "$WK/$NAME/extracted_images" "$S" "$E" <<'PY'
import os, sys
fr, dst, s, e = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
for i, f in enumerate(range(s, e)):
    os.symlink(f'{fr}/{f:06d}.jpg', f'{dst}/{i:04d}.jpg')
PY
  ln -sfn "$VID" "$WK/$NAME.mp4"          # names the sequence folder; frames come from extracted_images
  T0=$(date +%s)
  python $D/run_hawor_clip.py --video "$WK/$NAME.mp4" --calib "$CAL" --clip "${NAME}_raw" \
    --out "$WK/$NAME.out" --check-roundtrip < /dev/null > "$WK/$NAME.log" 2>&1 || { tail -20 "$WK/$NAME.log"; exit 1; }
  rm -f "$WK/$NAME"/tracks_*/model_masks.npy                          # ~2 GB per window
  echo "  $NAME [$S,$E) $(( ($(date +%s)-T0)/60 )) min | $(grep -a 'rows (' "$WK/$NAME.log" | tail -1 | sed 's/^.*\] *//')"
done

say "merge windows"
python $D/long/merge_windows.py --windows "$WK/windows.json" --work "$WK" \
  --out "$SCRATCH/${BASE}_raw_hand21_keypoints.npz" --sibling "$OUT/${C}_3d_keypoints.npz" \
  --meta-out "$OUT/${C}_hawor_metadata.json"

say "2/5 post-process (drop fabricated, smooth keypoints; no head pose)"
python $D/postprocess_v3.py --npz "$SCRATCH/${BASE}_raw_hand21_keypoints.npz" --out "$SCRATCH/pp.npz" 2>&1 | tail -25

say "3/5 MediaPipe second opinion (isolated env)"
conda activate mp310
[ -s "$SCRATCH/mp.npz" ] || python $D/mp_handedness.py --video "$VID" --out "$SCRATCH/mp.npz" 2>&1 | tail -8

say "4/5 per-hand handedness vote"
conda activate hawor
python $D/fix_handedness.py --npz "$SCRATCH/pp.npz" --out "$OUT/${C}_hand21_keypoints.npz" --mp "$SCRATCH/mp.npz" 2>&1 | tail -25
cp "$SCRATCH/pp_v3_report.json" "$OUT/${C}_v3_report.json" 2>/dev/null || true
mv -f "$OUT/${C}_hand21_keypoints_handedness_report.json" "$OUT/${C}_handedness_report.json" 2>/dev/null || true

say "4b/5 principal-point correction (HaWoR assumes the image centre)"
python $D/long/pp_correct.py --npz "$OUT/${C}_hand21_keypoints.npz"

say "5/5 QC + gate"
python ~/hawor_runs/scripts/qc_hawor.py --npz "$OUT/${C}_hand21_keypoints.npz" \
  --video "$VID" --clip "$C" --out "$OUT/qc" | tail -30
python - "$OUT" "$C" <<'PY'
import json,sys
o,c=sys.argv[1:3]
q=json.load(open(f'{o}/qc/{c}_qc.json'))
r=q['reproj']['median_px']
assert r < 1.0, f"FAIL reproj {r} px"
print(f"  GATE reproj {r:.4f} px OK | behind-lens {q['depth']['frac_any_joint_behind_lens']*100:.2f}%"
      f" | boneCV {q['bones_observed_only']['bone_cv_median']:.5f} | rows {q['rows']['total']}")
PY
rm -rf "$FR"                                                           # ~5 GB of JPEGs
say "V3-LONG-DONE"
