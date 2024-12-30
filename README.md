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
