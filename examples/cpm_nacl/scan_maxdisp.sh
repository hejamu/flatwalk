#!/bin/bash
# Acceptance vs step-size scan for the NaCl contract. Run on narvi:
#   nohup bash scan_maxdisp.sh > scan.log 2>&1 &
set -u
cd /home/ctn2867/WORK/playground/flatwalk-cpm
PY=.venv/bin/python
SCRIPT=torch_CPM/examples/flatwalk_cpm/nacl_contract.py
for d in 0.05 0.10 0.15 0.20 0.30; do
  echo "=== max_disp=$d ==="
  OMP_NUM_THREADS=4 "$PY" -u "$SCRIPT" \
    --plain-mc 4000 --nps 6 --n-pairs 32 --max-disp "$d" \
    --save-every 1000 --out "/tmp/scan_${d}.npz"
done
echo SCAN_DONE
