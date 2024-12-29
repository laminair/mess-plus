#!/bin/bash
#SBATCH -p lrz-hgx-h100-92x4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH -o output_%A.out
#SBATCH --exclude lrz-hgx-h100-030

#### cd "$(dirname "$0")"

export MASTER_PORT=13567
export WORLD_SIZE=8

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

srun --container-image='/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/Energy-Optimal-Inferencing/eoinference_v5.sqsh' \
     --container-mounts=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/:/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/,/dss/dsshome1/09/ge56heh2/:/dss/dsshome1/09/ge56heh2/,/sbin:/sbin,/usr/share/:/usr/share/ \
     --container-env=HF_TOKEN_PATH=/dss/dsshome1/09/ge56heh2/.cache/huggingface/token,/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/hf/misc,/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/datasets,HF_DATASETS_TRUST_REMOTE_CODE=True \
     --ntasks-per-node=4 \
     /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/Energy-Optimal-Inferencing/venv/bin/accelerate launch \
     --multi_gpu \
     --num_machines 2 \
     --num_processes 4 \
     --main_process_ip "$MASTER_ADDR" \
     --main_process_port $MASTER_PORT \
     --machine_rank 0 \
     --rdzv_backend static \
     --max_restarts 0 \
     lm_eval \
     --model hf \
     --model_args pretrained=meta-llama/Llama-3.1-405B-Instruct-FP8,trust_remote_code=True,parallelize=True,offload_buffers=True \
     --tasks logiqa2,boolq,mmlu \
     --batch_size auto:4 \
     --trust_remote_code \
     --log_samples \
     --output_path /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/Energy-Optimal-Inferencing/lm_eval_results


srun --container-image='/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/Energy-Optimal-Inferencing/eoinference_v5.sqsh' \
     --container-mounts=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/:/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/,/dss/dsshome1/09/ge56heh2/:/dss/dsshome1/09/ge56heh2/,/sbin:/sbin,/usr/share/:/usr/share/ \
     --container-env=HF_TOKEN_PATH=/dss/dsshome1/09/ge56heh2/.cache/huggingface/token,/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/hf/misc,/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/datasets,HF_DATASETS_TRUST_REMOTE_CODE=True \
     --ntasks-per-node=4 \
     /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/Energy-Optimal-Inferencing/venv/bin/accelerate launch \
     --multi_gpu \
     --num_machines 2 \
     --num_processes 4 \
     --main_process_ip "$MASTER_ADDR" \
     --main_process_port $MASTER_PORT \
     --machine_rank 1 \
     --rdzv_backend static \
     --max_restarts 0 \
     lm_eval \
     --model hf \
     --model_args pretrained=meta-llama/Llama-3.1-405B-Instruct-FP8,trust_remote_code=True,parallelize=True,offload_buffers=True \
     --tasks logiqa2,boolq,mmlu \
     --batch_size auto:4 \
     --trust_remote_code \
     --log_samples \
     --output_path /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/Energy-Optimal-Inferencing/lm_eval_results
