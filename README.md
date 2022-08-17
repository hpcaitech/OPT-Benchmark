# OPT-Benchmark

This benchmark is to compare the performance of Colossal-AI and DeepSpeed in terms of its zero redundancy optimizer and offloading. The script is adapted from the Hugging Face [example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling).


## Run Benchmarking

First, you need to install the following libraries.

```
# assuming using cuda 11.3
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install colossalai==0.1.9+torch1.11cu11.3 -f https://release.colossalai.org
pip install accelerate==0.10.0, datasets==1.18.4, transformers==4.21.0, deepspeed==0.6.5 tqdm
```

To run the benchmarking with different acceleration libraries, you can just execute the following bash script on a single node. We recommend you to run the `run_opt_clm.sh` script with one GPU first so as to download all the necessary files from Hugging Face.

```bash
# run with deepspeed zero 3 + offloading
bash ./run_gpt_clm.sh

# run with the current version of colossal-ai zero module
bash ./run_opt_clm_colossalai.sh

# run with the newer (experimental) version of colossal-ai zero module
bash ./run_opt_clm_colossalai_new.sh
```

Each script has 4 arguments.
- `BS`: batch size per GPU
- `MEMCAP`: whether to limit the GPU memory usage. For example, if MEMCAP = 40, the program will only use 40 GB memory even if the GPU has 80 GB. If MEMCAP=0, there is limit on the available GPU memory. The default value is 0.
- `MODEL`: the variant of the OPT MODEL, default is `13B`.
- GPUNUM: the number of GPUs to use, default is 8.

