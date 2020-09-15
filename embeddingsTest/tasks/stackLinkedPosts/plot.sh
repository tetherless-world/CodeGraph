#!/bin/bash

TYPE=$1

awk '/[0-9.]+/ { match($0, /([0-9.]+)/, a); print(a[1]); }' /tmp/true_${TYPE}.json > true.dat

awk '/[0-9.]+/ { match($0, /([0-9.]+)/, a); print(a[1]); }' /tmp/false_${TYPE}.json > false.dat

gnuplot distances.gnuplot > links.pdf
