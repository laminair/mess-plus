# Energy-Optimal-Inferencing


## Setup procedure

We use the `nvcr.io/nvidia/pytorch:24.09-py3` container for all of our experiments. 
The container is available from the [Nvidia NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags).

```
git clone https://github.com/ryanzhangofficial/Energy-Optimal-Inferencing.git
git submodule init 
git submodule update --init --recursive

python3.10 -m venv venv 
source venv/bin/activate
pip install -e lm-eval-harness/
pip install -r requirements.txt
```

You are good to go.

## Running MESS+ Experiments

Pick a configuration file from the `config/messplus/` folder and run the following command:
```commandline
VLLM_USE_V1=0 python3 mess_plus.py --config config/messplus/boolq.yaml --project-name mess-plus-benchmarks-v1
```
You can provide a `--project-name` of your choice. 
Make sure to check the GPU utilization settings and adjust them per your individual setup.

**Important note:** vLLM's new memory management cannot load two models into the same GPU. Therefore, we need to use 
`VLLM_USE_V1=0` to run experiments in a single-GPU environment.