#!/bin/bash

awk -f mrr-map.awk -v start=103 /tmp/out_usage_USE > mrr-map-use.dat
awk -f mrr-map.awk -v start=103 /tmp/out_usage_bert > mrr-map-bert.dat
awk -f mrr-map.awk -v start=103 /tmp/out_usage_roberta > mrr-map-roberta.dat

gnuplot mrr-map.gnuplot > mrr-map.pdf
