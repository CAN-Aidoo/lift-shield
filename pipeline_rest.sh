#!/bin/bash
cd /home/claude/lift
until grep -q TRAINDONE /tmp/claude-0/-home-claude/1990b9fd-8d5a-556e-9277-5349d8730891/tasks/b058wcw14.output 2>/dev/null; do sleep 20; done
NPROC=1 python controlled.py ALL_TRAIN > logs/controlled_train.txt 2>&1
until grep -q ENVDONE /tmp/claude-0/-home-claude/1990b9fd-8d5a-556e-9277-5349d8730891/tasks/b4bl3m4jt.output 2>/dev/null; do sleep 20; done
NPROC=2 python extended.py tightness > logs/tight.txt 2>&1
NPROC=2 python extended.py safetymap > logs/safety.txt 2>&1
echo PIPEDONE
