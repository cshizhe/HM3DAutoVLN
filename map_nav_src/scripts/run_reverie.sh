#! /bin/bash

DATA_ROOT=../datasets

train_alg=dagger

features=timm_vitb16
ft_dim=768
obj_features=timm_vitb16
obj_ft_dim=768

ngpus=1
seed=0

name=aug.hm3d.pseudo3d.depth2-${train_alg}-${features}-seed.${seed}-init.35k
outdir=${DATA_ROOT}/REVERIE/expr_duet/finetune_hm3d_v1/${name}

flag="--root_dir ${DATA_ROOT}
      --dataset reverie
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert

      --enc_full_graph
      --graph_sprels
      --fusion dynamic
      --multi_endpoints

      --dagger_sample sample

      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 100
      --max_objects 50

      --batch_size 8
      --lr 1e-5
      --iters 200000
      --log_every 1000
      --optim adamW

      --features ${features}
      --obj_features ${obj_features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --ml_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0."


# training with aug
python reverie/main_nav_obj_hm3d.py $flag --tokenizer bert \
      --bert_ckpt_file ${DATA_ROOT}/REVERIE/expr_duet/pretrain_hm3d_v1/pseudo3d-depth2-cmt-timm.vitb16-mlm.sap.og-init.lxmert-bsz.64/ckpts/model_step_35000.pt \
      --aug ${DATA_ROOT}/REVERIE/expr_speakers/2d_gpt2_prefix4_layer2/preds/ade20k_pseudo3d_depth2_epoch_94_beam0.jsonl \
      --eval_first 

# test
python reverie/main_nav_obj_hm3d.py $flag --tokenizer bert \
      --resume_file ${outdir}/ckpts/best_val_unseen \
      --test --submit 