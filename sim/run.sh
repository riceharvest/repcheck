#!/bin/bash
# usage: run.sh <jobs-file>   each line: <outname> <${SCRIPT:-central.py} args...>
cd ~/swarm-rl-sim
export CUDA_VISIBLE_DEVICES="" SIM_THREADS=${THR:-4}
grep -v '^#' "$1" | xargs -P ${PAR:-4} -L 1 env SCRIPT=${SCRIPT:-central.py} bash -c 'n=$0; shift 0; nice -n 10 .venv/bin/python ${SCRIPT:-central.py} "$@" --out results/$n.jsonl > results/$n.log 2>&1; echo done $n'
