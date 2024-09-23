#! /usr/bin/bash
# Time-stamp: <2024-09-23 11:02:19 rytis>
# Audacity label files do not support comments!!!
for i in 1 2 3 4 5
do
    awk 'BEGIN{n=0}{if (NR>1) {n+=1; printf "%.3f\t%.3f\t%d\n", $1, $1, n;}}' input/${i}.txt > input/${i}_label.txt
done


