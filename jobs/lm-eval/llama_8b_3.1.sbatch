#!/bin/bash
#SBATCH -p lrz-hgx-h100-92x4
#SBATCH --gres=gpu:2
#SBATCH --time=01:45:00
#SBATCH -o output_%A.out

cd "$(dirname "$0")"

nvidia-smi
enroot create /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/Energy-Optimal-Inferencing/eoinference_v5.sqsh

# Train
enroot start \
    --root \
    -e HF_TOKEN_PATH=/dss/dsshome1/09/ge56heh2/.cache/huggingface/token \
    -e HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/hf/misc \
    -e HF_DATASETS_CACHE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/datasets \
    --mount /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/ \
    --mount /dss/dsshome1/09/ge56heh2/ \
    --mount /sbin \
    --mount /usr/share/ \
    eoinference_v5 \
    /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/Energy-Optimal-Inferencing/venv/bin/accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    -m lm_eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,trust_remote_code=True,parallelize=True \
    --tasks logiqa2,boolq,mmlu \
    --device cuda:0 \
    --batch_size auto:4 \
    --trust_remote_code \
    --log_samples \
    --limit 20 \
    --output_path /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/Energy-Optimal-Inferencing/lm_eval_results
