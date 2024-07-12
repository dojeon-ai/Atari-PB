cd ..
cd ..
python run_pretrain.py \
    --config_name pretrain \
    --overrides group_name='dt' \
    --overrides ++dataloader.batch_size=64 \
    --overrides ++dataloader.t_step=8 \
    --overrides ++dataloader.n_step=0 \
    --overrides ++dataloader.gamma=1.0 \
    --overrides ++model.dt_embed_dim=512 \
    --overrides ++model.dt_mlp_dim=2048 \
    --overrides model/neck=dt_neck \
    --overrides model/head=mh_linear \
    --overrides ++trainer.type='dt' \
    --overrides ++trainer.rtg_scale=0.01 \
    --overrides ++trainer.num_epochs=12 \
    --overrides ++trainer.optimizer.lr=0.0001 \
    --overrides ++trainer.optimizer.weight_decay=0.05 \
    --overrides ++trainer.optimizer.betas=[0.9,0.95] \
    --overrides ++trainer.optimizer.eps=0.00000001 \