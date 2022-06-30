BS=32
colossalai run --nproc_per_node 1 \
    --master_port 29600 \
    run_clm_no_trainer_colossalai.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path facebook/opt-6.7b \
    --output_dir /tmp/test-clm \
    --per_device_train_batch_size ${BS} 2>&1 | tee ./logs/colo_bs_${BS}.log
