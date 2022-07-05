python train_wandb.py \
	--debug=True\
	--debug_train_num=10\
	--debug_valid_num=2\
	--train_batch=16\
	--valid_batch=64\
	--model_path=bert-base-chinese \
	--learning_rate=5e-5 \
	--train_rate=0.8 \
	--content_size=100 \
	--epoch_number=1 \
	--freeze=True \
	--loss=MCE \
	--optm=Adam \
	--pn_rate=1 \
	--class_num=234 \
	--time_limit=30 \
	--f1_limit=0.55 \
	--diff_limit=2 \
	--f1_save_limit=0.3 

sleep 2

python train_wandb.py \
	--debug=True\
	--debug_train_num=10\
	--debug_valid_num=2\
	--train_batch=16\
	--valid_batch=64\
	--model_path=bert-base-chinese \
	--learning_rate=5e-5 \
	--train_rate=0.8 \
	--content_size=100 \
	--epoch_number=1 \
	--freeze=True \
	--loss=MCE \
	--optm=Adam \
	--pn_rate=1 \
	--class_num=234 \
	--time_limit=30 \
	--f1_limit=0.55 \
	--diff_limit=2 \
	--f1_save_limit=0.3 

