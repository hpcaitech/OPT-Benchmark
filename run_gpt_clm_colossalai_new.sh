colossalai run --nproc_per_node 4 \
    --master_port 29600 \
    run_clm_no_trainer_colossalai_new.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path facebook/opt-1.3b \
    --output_dir /tmp/test-clm \
    --per_device_train_batch_size 72