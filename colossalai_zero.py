from colossalai.zero.shard_utils import TensorShardStrategy

zero = dict(model_config=dict(shard_strategy=TensorShardStrategy(),
                              tensor_placement_policy="auto",
                              reuse_fp16_shard=True,
                              warmup_non_model_data_ratio=0.9),
            optimizer_config=dict(gpu_margin_mem_ratio=0.8))
