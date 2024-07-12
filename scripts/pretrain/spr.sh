cd ..
cd ..
python run_pretrain.py \
    --config_name pretrain \
    --overrides group_name='spr' \
    --overrides ++dataloader.batch_size=128 \
    --overrides ++dataloader.t_step=5 \
    --overrides ++dataloader.n_step=0 \
    --overrides model/head=spr_head \
    --overrides ++trainer.type='spr' \
    --overrides ++trainer.tau_scheduler='{initial_value:0.99,final_value:0.999,max_step:-1}' \
    --overrides ++trainer.target_update_every=1 \
    --overrides trainer.num_epochs=25 \
    --overrides trainer.save_every=5 \
    --overrides trainer.eval_every=5 \
    --overrides ++trainer.optimizer.lr=0.00003 \
    --overrides ++trainer.optimizer.weight_decay=0.00001 \
    --overrides ++trainer.optimizer.betas=[0.9,0.999] \
    --overrides ++trainer.optimizer.eps=0.00000001