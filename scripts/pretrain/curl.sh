# head.action_size(output dim) = neck.hidden_dims[-1]
cd ..
cd ..
python run_pretrain.py \
    --config_name pretrain \
    --overrides group_name='curl' \
    --overrides ++model.head.action_size=512 \
    --overrides ++trainer.type='curl' \
    --overrides ++trainer.temperature=1.0 \
    --overrides ++trainer.tau_scheduler='{initial_value:0.99,final_value:0.999,max_step:-1}' \
    --overrides ++trainer.target_update_every=1 \
    --overrides ++trainer.optimizer.lr=0.000003 \
    --overrides ++trainer.optimizer.weight_decay=0.00001 \
    --overrides ++trainer.optimizer.betas=[0.9,0.999] \
    --overrides ++trainer.optimizer.eps=0.00000001