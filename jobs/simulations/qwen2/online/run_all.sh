#!/bin/bash

sbatch arc_challenge.sbatch
sleep 1
sbatch arc_easy.sbatch
sleep 1
sbatch boolq.sbatch
sleep 1
sbatch lambada_standard.sbatch
sleep 1
sbatch logiqa.sbatch
sleep 1
sbatch logiqa2.sbatch
sleep 1
sbatch piqa.sbatch
sleep 1
sbatch sciq.sbatch
sleep 1
sbatch social_iqa.sbatch
sleep 1
 sbatch winogrande.sbatch
 sleep 1
echo "All jobs launched!"
