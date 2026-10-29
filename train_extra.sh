#!/bin/bash
# Addendum A training rows T1-T4 (B2 = residual, B5 = residualS), 5 seeds each
cd /home/claude/lift; mkdir -p logs
run() { for s in 0 1 2 3 4; do for mode in residual residualS; do env "$@" python train_es.py $mode $s > logs/train_${mode}_s${s}_$(echo "$@" | tr ' =' '__').txt 2>&1; done; done; }
export -f run 2>/dev/null
( run TAG=_T1 R_HI=1.0 ) &
( run TAG=_T2 ST=1.5 ) &
wait
( run TAG=_T3 LAM=10 ) &
( run TAG=_T4 ITERS=40 ) &
wait
echo TRAINDONE
