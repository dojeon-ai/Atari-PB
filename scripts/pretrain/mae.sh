cd ..
cd ..
python run_pretrain.py \
    --config_name pretrain \
    --overrides group_name='mae' \
    --overrides ++model.mask_ratio=0.9 \
    --overrides ++model.vit_embed_dim=512 \
    --overrides ++model.vit_mlp_dim=2048 \
    --overrides model/neck=mae_neck \
    --overrides model/head=mae_head \
    --overrides ++trainer.type='mae' \
    --overrides ++trainer.clip_grad_norm=null \
    --overrides ++trainer.optimizer.lr=0.0003 \
    --overrides ++trainer.optimizer.weight_decay=0.05 \
    --overrides ++trainer.optimizer.betas=[0.9,0.95] \
    --overrides ++trainer.optimizer.eps=0.00000001


