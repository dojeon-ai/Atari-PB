# head.in_shape = neck.hidden_dims[-1]*2
cd ..
cd ..
python run_pretrain.py \
    --config_name pretrain \
    --overrides group_name='idm' \
    --overrides ++dataloader.dataset_type='random_future_state' \
    --overrides ++dataloader.n_step=1 \
    --overrides ++model.head.in_shape=1024 \
    --overrides ++trainer.type='idm' \
    --overrides ++trainer.optimizer.lr=0.0003 \
    --overrides ++trainer.optimizer.weight_decay=0.00001 \
    --overrides ++trainer.optimizer.betas=[0.9,0.999] \
    --overrides ++trainer.optimizer.eps=0.00000001