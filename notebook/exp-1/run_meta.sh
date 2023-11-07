#!/bin/bash
for EPOCH in 6 7 8 9 10
do
    sbatch run.sh $EPOCH
    sleep 1
done