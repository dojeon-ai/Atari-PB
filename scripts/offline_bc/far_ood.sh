cd ..
cd ..
python run_offline_bc.py \
    --config_name offline_bc \
    --overrides group_name=test \
    --overrides exp_name=test \
    --overrides dataloader=far_ood_offline_bc \
    --overrides num_gpus_per_node=2 \
    --overrides num_exps_per_gpu=2 \
    --overrides pretrained=load_b_freeze_b \
    --overrides pretrained.ckpt_path='your_path_here' \