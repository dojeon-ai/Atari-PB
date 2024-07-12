cd ..
cd ..
python run_zeroshot.py \
    --config_name zero_shot \
    --overrides exp_name='dt_rtg1' \
    --overrides ++num_gpus_per_node=2 \
    --overrides ++num_exps_per_gpu=2 \
    --overrides ++dataloader.t_step=8 \
    --overrides ++dataloader.n_step=0 \
    --overrides ++dataloader.gamma=1.0 \
    --overrides ++model.dt_embed_dim=512 \
    --overrides ++model.dt_mlp_dim=2048 \
    --overrides model/neck=dt_neck \
    --overrides model/head=mh_linear \
    --overrides ++trainer.type='dt' \
    --overrides ++trainer.rtg_scale=0.01 \
    --overrides pretrained.ckpt_path='your_path_here' \