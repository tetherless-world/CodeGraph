#!/bin/bash

export PYTHONPATH=../:.

# run ranking task
sh run_ranking_task.sh

# run correlation task
sh run_correlation_task.sh
