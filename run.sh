python main.py \
	--run_name "[ablation] add parameter noise" \
	--fcnet_hiddens 256 1024 4096 \
	--lr 0.00005 \
	--minibatch_size 512 \
	--num_sgd_iter 10 \
	--train_batch_size 3200 \
	--num_iterations 140 \
	--checkpoint_every 20