cd ..
cd ..
python run_pretrain.py \
    --config_name pretrain \
    --overrides group_name='siammae' \
    --overrides ++dataloader.dataset_type='random_future_state' \
    --overrides ++model.mask_ratio=0.95 \
    --overrides ++model.vit_embed_dim=512 \
    --overrides ++model.vit_mlp_dim=2048 \
    --overrides model/neck=siammae_neck \
    --overrides model/head=siammae_head \
    --overrides ++trainer.type='siammae' \
    --overrides ++trainer.clip_grad_norm=null \
    --overrides ++trainer.optimizer.lr=0.0003 \
    --overrides ++trainer.optimizer.weight_decay=0.05 \
    --overrides ++trainer.optimizer.betas=[0.9,0.95] \
    --overrides ++trainer.optimizer.eps=0.00000001
