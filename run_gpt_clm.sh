export BS=${BS:-16}
export MEMCAP=${MEMCAP:-40}
export MODEL=${MODEL:-"6.7b"}
export GPUNUM=${GPUNUM:-1}

torchrun --nproc_per_node ${GPUNUM} \
    --standalone \
    run_clm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path facebook/opt-${MODEL} \
    --output_dir /tmp/test-clm \
    --mem_cap ${MEMCAP} \
    --per_device_train_batch_size ${BS} 2>&1 | tee logs/ds_${MODEL}_bs_${BS}_cap_${MEMCAP}_gpu_${GPUNUM}.log




