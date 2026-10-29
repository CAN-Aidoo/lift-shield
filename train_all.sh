#!/bin/bash
cd /home/claude/lift
for s in 0 1 2 3 4; do
  for mode in residual residualS; do python train_es.py $mode $s > weights_log_${mode}_s$s.txt 2>&1 & done
done
wait
echo ALLDONE
