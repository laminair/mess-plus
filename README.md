# [NeurIPS 2025] Dynamically Learned Test-Time Model Routing in Language Model Zoos with Service Level Guarantees

[![arXiv](https://img.shields.io/badge/arXiv-2505.19947-b31b1b.svg)](https://arxiv.org/abs/2505.19947)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)


## Getting started 
We use `uv` to maintain dependencies. You can install it by following the guide here: [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/). 

Setting up the project is straight forward: 
```
git clone https://github.com/laminair/mess-plus.git
cd mess-plus 
uv sync
```

And you are ready to go. 
This project uses Weights & Biases for logging. You may have to configure your machine to use W&B before launching any experiment. 
Note that we have used enroot and the `nvcr.io/nvidia/pytorch:24.09-py3` container for our experiments to improve reproducibility.
The container is available from the [Nvidia NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags).

## Running experiments
We have divided the setup for our paper experiments into two major parts: 
1. Data capturing (since inference at scale is time-consuming and expensive)
2. Evaluation of MESS+ (for fast iteration and prototyping)

**Important note:** You can also run MESS+ directly when doing inference. This is useful when you want to explore our code base. This is important for the subsequent configuration 


### 1. Data capturing 
You can run MESS+ with any model zoo you want. To do so, pick a config `yaml` file from any of the sub-folders and modify it. 
Here are some important notes: 
- You must order the models in increasing order by operating cost (for local inference that is likely determined by the parameter size)
- You must specify the GPU where the model should run (note: models are deployed in parallel!)
- You can specify a quantization mechanism for faster inference (if the model config supports it)
- The classifier model is set to `ModernBert` ([HuggingFace](https://huggingface.co/blog/modernbert)) but you may choose any model that you like. This can be directly adjusted in the config file.
- There are two variables where you can set alpha: `alpha` and `alpha_values`. `alpha` is used in the online inference script and `alpha_values` in the simulator.
- The LM Eval Harness config can be used to store the inference results in a csv file and directly use it for the MESS+ evaluation. This can be done by setting `write_to_disk` to `true`. 

### 2. Evaluation of MESS+ 
You can use any pre-generated inference outputs for evaluation of MESS+. The `mess_plus_simulator.py` script expects input data of shape: 
```
doc_id,input_text,benchmark_name,label_xsmall,acc_norm_xsmall,energy_consumption_xsmall,inference_time_xsmall,label_small,acc_norm_small,energy_consumption_small,inference_time_small,label_medium,acc_norm_medium,energy_consumption_medium,inference_time_medium,label_large,acc_norm_large,energy_consumption_large,inference_time_large
```

### 3. Launching an experiment
For all scenarios we provide `slurm` scripts with the full execution commands. Please refer to those files for running experiments.

Note that `acc_norm` can be replaced by `acc` for benchmarks where no normalized accuracy is available. 
Also the model label (e.g., `xsmall`) must match the labels in your yaml config file.  


## Known technical issues
`VLLM` has a V1 and V2 engine where the V2 engine's memory management does not work well when deploying multiple model onto a single GPU. 
You can get around this by specifying `VLLM_USE_V1=0 python3 mess_plus.py ...`.
The `Zeus` energy monitoring library sometimes recognizes CPUs to have RAPL capabilities without them actually having them. 
This can lead to an error. To solve this, disable RAPL at startup and monitor the GPUs only.
