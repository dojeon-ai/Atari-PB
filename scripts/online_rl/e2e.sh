cd ..
cd ..
python run_online_rl.py \
    --config_name online_rl \
    --overrides group_name=test \
    --overrides exp_name=test \
    --overrides num_gpus_per_node=2 \
    --overrides num_exps_per_gpu=2 \
    --overrides pretrained=load_none_freeze_none \