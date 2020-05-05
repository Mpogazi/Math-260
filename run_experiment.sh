#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Run with ./run_experiments [experiment_number]."
    exit 1
fi

export PYTHONPATH=$PYTHONPATH:src/
python experiments/exp$1.py
