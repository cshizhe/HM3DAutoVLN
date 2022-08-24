# HM3D-AutoVLN: Automatic VLN Data Generation

## Building-level Pre-processing
1. install [mask2former](https://github.com/facebookresearch/Mask2Former).

2. download [HM3D scenes](https://github.com/matterport/habitat-matterport-3dresearch).

3. generate navigation graphs and pseudo 3d object labels
```
cd hm3d_data_gen

python step00_create_hm3d_nav_graphs.py ../datasets/HM3D/nav_graphs_v1/connectivity

python step01_create_view_images.py

python step02_extract_view_features.py --model_name vit_base_patch16_224 \
    --output_dir ../datasets/HM3D/nav_graphs_v1/features/view_timm_imagenet_vitb16

python step03_detect_obj2d_ade20k.py \
    --output_dir ../datasets/HM3D/nav_graphs_v1/features/obj2d_ade20k

python step04_obj_2d_to_3d.py
python step05_clean_obj2d_by_3d.py

python step06_extract_obj_features.py --model_name vit_base_patch16_224 \
    --output_dir $SCRATCH/datasets/HM3D/nav_graphs_v1/features/obj2d_ade20k_pseudo3d_merged_timm_imagenet_vitb16
```

## Generating VLN Training Triplets
1. train a speaker model on REVERIE dataset or download our [trained speaker model]().
```
cd reverie_speaker_src
python train_gt_obj.py --config_file configs/gpt2cap_2d_config.yaml
```

2. inference
```
outdir=../datasets/REVERIE/expr_speakers/2d_gpt2_prefix4_layer2
# evaluate
python train_gt_obj.py --config_file $outdir/logs/config.yaml --eval val_unseen --resume_epoch 94
# inference
python gen_hm3d_captions_bbox2d.py --config_file $outdir/logs/config.yaml \
    --resume_epoch 94 --batch_size 64 
```