cd ..
cd ..

overrides=(
   group_name='cam_video'
   save_type='video'
   +use_blank=False
   +use_all=True
   dataloader.batch_size=1
   dataloader.num_workers=0
   dataloader.games=['breakout']
   dataloader.samples_per_checkpoint=1000
   dataloader.shuffle=False
)

model_overrides=(
   +model@model.MAE=mae
   +model@model.CURL=curl
   +model@model.SiamMAE=siammae
   +model@model.BC=bc
   +model@model.IDM=idm
   +model@model.DT=dt
   +model@model.CQL_D=distributional
)

pretrained_overrides=(
   +pretrained@pretrained.Random=load_none_freeze_b
   +pretrained@pretrained.MAE=load_b_freeze_b
   +pretrained@pretrained.CURL=load_b_freeze_b
   +pretrained@pretrained.ATC=load_b_freeze_b
   +pretrained@pretrained.SiamMAE=load_b_freeze_b
   +pretrained@pretrained.R3M=load_b_freeze_b
   +pretrained@pretrained.BC=load_b_freeze_b
   +pretrained@pretrained.SPR=load_b_freeze_b
   +pretrained@pretrained.IDM=load_b_freeze_b
   +pretrained@pretrained.DT=load_b_freeze_b
   +pretrained@pretrained.CQL_M=load_b_freeze_b
   +pretrained@pretrained.CQL_D=load_b_freeze_b
)

ckpt_overrides=(
   pretrained.MAE.ckpt_path='mae_path'
   pretrained.CURL.ckpt_path='curl_path'
   pretrained.ATC.ckpt_path='atc_path'
   pretrained.SiamMAE.ckpt_path='siammae_path'
   pretrained.R3M.ckpt_path='r3m_path'
   pretrained.BC.ckpt_path='bc_path'
   pretrained.SPR.ckpt_path='spr_path'
   pretrained.IDM.ckpt_path='idm_path'
   pretrained.DT.ckpt_path='dt_path'
   pretrained.CQL_M.ckpt_path='cql_m_path'
   pretrained.CQL_D.ckpt_path='cql_d_path'
)

overrides+=("${model_overrides[@]}")
overrides+=("${pretrained_overrides[@]}")
overrides+=("${ckpt_overrides[@]}")

cmd="python run_offline_eigencam.py"
for override in "${overrides[@]}"; do
    cmd+=" --overrides $override"
done

eval $cmd
    