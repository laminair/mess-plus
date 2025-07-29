#!/bin/bash

sbatch arc_challenge.sbatch
sleep 1
sbatch winogrande.sbatch
sleep 1
echo "All jobs launched!"
