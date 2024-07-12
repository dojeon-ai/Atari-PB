cd ..
cd ..
python run_pretrain.py \
    --config_name pretrain \
    --overrides group_name='cql' \
    --overrides ++trainer.type='cql' \
    --overrides ++trainer.v_min=-10 \
    --overrides ++trainer.v_max=10 \
    --overrides ++trainer.num_atoms=51 \
    --overrides ++trainer.feature_normalization=False \
    --overrides ++trainer.cql_coefficient=0.1 \
    --overrides ++trainer.target_tau=0.99 \
    --overrides ++trainer.optimizer.lr=0.0001 \
    --overrides ++trainer.optimizer.weight_decay=0.00001 \
    --overrides ++trainer.optimizer.betas=[0.9,0.999] \
    --overrides ++trainer.optimizer.eps=0.00015