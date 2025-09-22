#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=12:30:00
#SBATCH -o output_%A.out

cd "$(dirname "$0")"

nvidia-smi
enroot create /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/messplus.sqsh

# Run the mess+ evaluation
enroot start \
    --root \
    -e HF_TOKEN_PATH=/dss/dsshome1/09/ge56heh2/.cache/huggingface/token \
    -e HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/hf/misc \
    -e HF_DATASETS_CACHE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/datasets \
    -e HF_DATASETS_TRUST_REMOTE_CODE=True \
    -e VLLM_USE_V1=0 \
    --mount /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/ \
    --mount /dss/dsshome1/09/ge56heh2/ \
    --mount /sbin \
    --mount /usr/share/ \
    messplus \
    /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/mess-plus/venv/bin/python /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/mess-plus/main.py \
    --config /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/mess-plus/config/mixed_l3_q2/logiqa.yaml \
    --wandb-entity tum-i13 \
    --wandb-project mess-plus_mixed-l3-q2_simulation_vFINAL \
    --dataset-path /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/mess-plus/data/mixed_l3_q2/inference_outputs/logiqa
