# run_ids: The ids of your wandb runs (e.g., 595h0sen)
# ckpt_path: Path to your checkpoint file (pth)

cd ..
cd ..
python run_pretrain.py \
    --config_name pretrain \
    --overrides group_name='bc' \
    --overrides ++trainer.optimizer.lr=0.0001 \
    --overrides ++trainer.optimizer.weight_decay=0.00001 \
    --overrides ++trainer.optimizer.betas=[0.9,0.999] \
    --overrides ++trainer.optimizer.eps=0.00000001 \
    --overrides trainer.resume=True \
    --overrides trainer.run_ids=[wandb_ids_here] \
    --overrides trainer.ckpt_path='ckpt_path_here'