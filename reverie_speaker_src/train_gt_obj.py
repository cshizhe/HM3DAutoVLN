import os
import argparse
from tqdm import tqdm
import numpy as np
import json

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from config import get_cfg_defaults

from gpt2cap_model import ReveriePanoObject2DGPT2CapModel
from gpt2cap_model import ReveriePanoObject2DLSTMCapModel
from dataset import ReveriePanoObject2DCaptionDataset, collate_fn
from utils import setup_seeds, setup_dirs, set_logger
from utils import Metrics, evaluate_caption

from transformers import AdamW, get_linear_schedule_with_warmup


def model_data_factory(cfg):
    if cfg.MODEL.TYPE == 'gpt2':
        model_class = ReveriePanoObject2DGPT2CapModel
    elif cfg.MODEL.TYPE == 'lstm':
        model_class = ReveriePanoObject2DLSTMCapModel
    else:
        raise NotImplementedError('incorrect model type: %s' % (cfg.MODEL.TYPE))

    data_class = ReveriePanoObject2DCaptionDataset
    data_collate_fn = collate_fn
    
    return model_class, data_class, data_collate_fn


def train(cfg):
    setup_seeds(cfg.SEED)
    log_dir, ckpt_dir, pred_dir = setup_dirs(cfg.OUTPUT_DIR)
    logger = set_logger(os.path.join(log_dir, 'train.log'))
    assert logger is not None

    with open(os.path.join(log_dir, 'config.yaml'), 'w') as outf:
        outf.write(cfg.dump())

    model_class, data_class, data_collate_fn = model_data_factory(cfg)
    
    cap_model = model_class(cfg.MODEL)
    if cfg.RESUME_FILE is not None:
        cap_model.load(cfg.RESUME_FILE)
    cap_model = cap_model.to(cap_model.device)
    print(cap_model)

    trn_dataset = data_class(
        cfg.DATA.VIEW_FT_DIR if cfg.MODEL.USE_VIEW_FT else None, 
        cfg.DATA.OBJ_FT_DIR, cfg.DATA.ANNO_DIR, 
        'train', view_ft_size=cfg.MODEL.VIEWIMG_FT_SIZE,
        obj_ft_size=cfg.MODEL.OBJIMG_FT_SIZE,
        use_ctx_objs=cfg.MODEL.USE_CTX_OBJS,
        image_height=480, image_width=640, max_txt_len=cfg.MODEL.MAX_TXT_LEN, 
        in_memory=True, is_train=True
    )
    trn_loader = DataLoader(
        trn_dataset, cfg.TRAIN.BATCH_SIZE, shuffle=True, 
        pin_memory=True, collate_fn=data_collate_fn, num_workers=0
    )

    val_dataset = data_class(
        cfg.DATA.VIEW_FT_DIR if cfg.MODEL.USE_VIEW_FT else None, 
        cfg.DATA.OBJ_FT_DIR, cfg.DATA.ANNO_DIR, 
        'val_unseen', view_ft_size=cfg.MODEL.VIEWIMG_FT_SIZE,
        obj_ft_size=cfg.MODEL.OBJIMG_FT_SIZE,
        use_ctx_objs=cfg.MODEL.USE_CTX_OBJS,
        image_height=480, image_width=640, max_txt_len=cfg.MODEL.MAX_TXT_LEN, 
        in_memory=True, is_train=False
    )
    val_loader = DataLoader(
        val_dataset, cfg.TRAIN.BATCH_SIZE, shuffle=False, 
        pin_memory=True, collate_fn=data_collate_fn, num_workers=0
    )

    optimizer = AdamW(cap_model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.TRAIN.WARMUP_STEPS, 
        num_training_steps=cfg.TRAIN.NUM_EPOCH * len(trn_loader)
    )
    print('steps/epoch=%d, epoch=%d, total training steps=%d' % (
        len(trn_loader), cfg.TRAIN.NUM_EPOCH, cfg.TRAIN.NUM_EPOCH * len(trn_loader))
    )

    sample_pred_keys = None
    for epoch in range(cfg.TRAIN.NUM_EPOCH):
        cap_model.train()
        loss_metric = Metrics()
        for batch in tqdm(trn_loader):
            optimizer.zero_grad()

            cap_logits, cap_loss = cap_model(batch, compute_loss=True)

            cap_loss.backward()
            optimizer.step()
            scheduler.step()

            loss_metric.accumulate(cap_loss.data.item())
        
        logger.info('train epoch %d avg_loss: %.4f, lr: %.6f' % (
            epoch, loss_metric.average, optimizer.param_groups[0]['lr'])
        )

        if (epoch + 1) % cfg.TRAIN.EVAL_EVERY_EPOCH == 0:
            cap_model.eval()
            loss_metric = Metrics()
            all_pred_caps, all_ref_caps = {}, {}
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    cap_logits, cap_loss = cap_model(batch, compute_loss=True)
                    loss_metric.accumulate(cap_loss.data.item())
                    pred_caps = cap_model.greedy_inference(batch)
                    for j, bid in enumerate(batch['names']):
                        all_pred_caps[bid] = [pred_caps[j]]
                        all_ref_caps[bid] = batch['ref_caps'][j]

            print('sample preds')
            if sample_pred_keys is None:
                sample_pred_keys = list(all_pred_caps.keys())
                sample_pred_keys = [sample_pred_keys[ridx] for ridx in \
                    np.random.permutation(len(all_pred_caps))[:10]]
            for spkey in sample_pred_keys:
                print(spkey, all_pred_caps[spkey][0])

            cap_scores = evaluate_caption(
                ref_caps=all_ref_caps, pred_caps=all_pred_caps,
                scorer_names=['bleu4', 'rouge', 'cider']
            )
            logger.info('\tval_unseen epoch %d avg_loss: %.4f' % (epoch, loss_metric.average))
            logger.info(', '.join(['%s:%.2f'%(k, v) for k, v in cap_scores.items()]))

            cap_model.save(os.path.join(ckpt_dir, f'epoch_{epoch}.pth'))

def validate(cfg, split, beam):
    log_dir, ckpt_dir, pred_dir = setup_dirs(cfg.OUTPUT_DIR)

    model_class, data_class, data_collate_fn = model_data_factory(cfg)
    
    cap_model = model_class(cfg.MODEL)
    assert cfg.RESUME_FILE is not None
    cap_model.load(cfg.RESUME_FILE)
    cap_model = cap_model.to(cap_model.device)

    val_dataset = data_class(
        cfg.DATA.VIEW_FT_DIR if cfg.MODEL.USE_VIEW_FT else None, 
        cfg.DATA.OBJ_FT_DIR, cfg.DATA.ANNO_DIR, 
        split, view_ft_size=cfg.MODEL.VIEWIMG_FT_SIZE,
        obj_ft_size=cfg.MODEL.OBJIMG_FT_SIZE, use_ctx_objs=cfg.MODEL.USE_CTX_OBJS,
        image_height=480, image_width=640, max_txt_len=cfg.MODEL.MAX_TXT_LEN, 
        in_memory=True, is_train=False
    )
    val_loader = DataLoader(
        val_dataset, cfg.TRAIN.BATCH_SIZE, shuffle=False, 
        pin_memory=True, collate_fn=data_collate_fn, num_workers=0
    )

    cap_model.eval()
    all_pred_caps, all_ref_caps = {}, {}
    with torch.no_grad():
        for batch in tqdm(val_loader):
            if beam == 0:
                pred_caps = cap_model.greedy_inference(batch)
                for j, bid in enumerate(batch['names']):
                    all_pred_caps[bid] = [pred_caps[j]]
                    all_ref_caps[bid] = batch['ref_caps'][j]
            else:
                pred_caps = cap_model.beam_inference(batch, beam_size=beam)
                for j, bid in enumerate(batch['names']):
                    all_pred_caps[bid] = [pred_caps[j][0][1]]
                    all_ref_caps[bid] = batch['ref_caps'][j]

    cap_scores = evaluate_caption(
        ref_caps=all_ref_caps, pred_caps=all_pred_caps,
        scorer_names=['bleu4', 'meteor', 'rouge', 'cider']
    )
    print(', '.join(['%s:%.2f'%(k, v) for k, v in cap_scores.items()]))

    model_name = os.path.basename(cfg.RESUME_FILE)[:-4]
    with open(os.path.join(pred_dir, '%s_%s_beam%d.json'%(split, model_name, beam)), 'w') as outf:
        json.dump({k: v[0] for k, v in all_pred_caps.items()}, outf, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='configs/gpt2cap_config.yaml')
    parser.add_argument('--eval', default=None, choices=['val_seen', 'val_unseen'])
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--beam', type=int, default=0)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    if args.eval:
        cfg.RESUME_FILE = os.path.join(cfg.OUTPUT_DIR, 'ckpts', 'epoch_%d.pth'%args.resume_epoch)
    cfg.freeze()

    if args.eval is not None:
        validate(cfg, args.eval, args.beam)
    else:
        train(cfg)


if __name__ == '__main__':
    main()
        
