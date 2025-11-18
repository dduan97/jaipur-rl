python evaluate.py \
	--chkpt_dir "/home/ubuntu/cs230/checkpoints/[hero]-3-hiddens-less-checkpointing_lr5e-05_mbs512_sgditer10_tbs3200_hiddens256-1024-4096-_intermediaterewardsFalse/" \
    --min_step 20 \
    --max_step 701 \
    --interval_step 20 \
    --num_eval_episodes 100 \
    --benchmark_agent random 