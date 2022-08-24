import os
import argparse
from tqdm import tqdm
import numpy as np
import json
import jsonlines
import collections
import glob
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from config import get_cfg_defaults
from dataset import calc_angle_feature, ReveriePanoObject2DCaptionDataset
from dataset import collate_fn
from utils import setup_seeds, setup_dirs
from train_gt_obj import model_data_factory

hm3d_dir = '../datasets/HM3D'
nav_graph_dir = os.path.join(hm3d_dir, 'nav_graphs_v1')
connectivity_dir = os.path.join(nav_graph_dir, 'connectivity')


class HM3DPanoCaptionDataset(ReveriePanoObject2DCaptionDataset):
    def __init__(
        self, view_ft_dir, obj_ft_dir, obj_det_dir, anno_dir, scans
    ):
        super().__init__(
            view_ft_dir, obj_ft_dir, anno_dir, None,
            view_ft_size=768, obj_ft_size=768, 
            image_height=224, image_width=224,
            in_memory=False, max_txt_len=100, is_train=False
        )
        self.data_ids = []
        for scan in scans:
            with jsonlines.open(os.path.join(obj_det_dir, scan, 'view_bboxes_merged_by_3d.jsonl'), 'r') as f:
                for item in f:
                    for i, x in enumerate(item['bboxes']):
                        self.data_ids.append((item['scanvp'], x['obj_id']))

        self.objname_maps = {k: k for k in self.objname_fts.keys()}

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        scanvp, tgt_objid = self.data_ids[idx]
        scan, vp = scanvp.split('_')
        scanvpobj = '%s_%s_%s' % (scan, vp, str(tgt_objid))

        obj_data = self.get_obj_feature(scan, vp)
        # image ft, name glove42b, center angle, size
        obj_fts = np.zeros((len(obj_data), self.obj_ft_size + 300 + 4 + 3), dtype=np.float32)
        obj_fts[0, :self.obj_ft_size] = obj_data[tgt_objid]['img_ft']
        obj_fts[0, self.obj_ft_size: self.obj_ft_size+300] = obj_data[tgt_objid]['name_ft']
        obj_fts[0, self.obj_ft_size+300: self.obj_ft_size+304] = obj_data[tgt_objid]['center_ft']
        obj_fts[0, self.obj_ft_size+304: ] = obj_data[tgt_objid]['size_ft']

        k = 1
        for objid, value in obj_data.items():
            if str(objid) != str(tgt_objid):
                obj_fts[k, :self.obj_ft_size] = value['img_ft']
                obj_fts[k, self.obj_ft_size: self.obj_ft_size+300] = value['name_ft']
                obj_fts[k, self.obj_ft_size+300: self.obj_ft_size+304] = value['center_ft']
                obj_fts[k, self.obj_ft_size+304:] = value['size_ft']
                k += 1

        if self.view_ft_dir is not None:
            # image ft, angle ft
            view_fts = self.get_view_feature(scan, vp)
            view_fts = np.concatenate([view_fts, self.view_angle_features], 1)
            reorder_view_fts = np.zeros(view_fts.shape, dtype=np.float32)
            reorder_view_fts[0] = view_fts[obj_data[tgt_objid]['view_id']]
            k = 1
            for i in range(36):
                if i != obj_data[tgt_objid]['view_id']:
                    reorder_view_fts[k] = view_fts[i]
                    k += 1
            view_fts = reorder_view_fts
        else:
            view_fts = None

        return {
            'names': scanvpobj,
            'obj_fts': obj_fts,
            'view_fts': view_fts,
        }



def proc_generate_captions(proc_id, scans, beam, cfg):
    print('start proc_id %d: %d scans' % (proc_id, len(scans)))

    assert cfg.RESUME_FILE is not None
    log_dir, ckpt_dir, pred_dir = setup_dirs(cfg.OUTPUT_DIR)
    model_name = os.path.basename(cfg.RESUME_FILE)[:-4]

    # pred_dir = os.path.join(pred_dir, 'ade20k_obj2d_%s_beam%d'%(model_name, beam))
    pred_dir = os.path.join(pred_dir, 'ade20k_pseudo3d_%s_beam%d'%(model_name, beam))
    os.makedirs(pred_dir, exist_ok=True)

    model_class, _, _ = model_data_factory(cfg)
    data_class = HM3DPanoCaptionDataset
    data_collate_fn = collate_fn

    cap_model = model_class(cfg.MODEL)
    
    cap_model.load(cfg.RESUME_FILE)
    cap_model = cap_model.to(cap_model.device)
    cap_model.eval()
    torch.set_grad_enabled(False)

    view_ft_dir = os.path.join(nav_graph_dir, 'features', 'view_timm_imagenet_vitb16')
    obj_ft_dir = os.path.join(nav_graph_dir, 'features', 'obj2d_ade20k_pseudo3d_merged_timm_imagenet_vitb16')
    obj_det_dir = os.path.join(nav_graph_dir, 'features', 'obj2d_ade20k')
    anno_dir = '../datasets/REVERIE/annotations/speaker_inputs'

    for scan in tqdm(scans):
        if os.path.exists(os.path.join(pred_dir, '%s.jsonl'%scan)):
            continue
        st_time = time.time()
        
        val_dataset = data_class(
            view_ft_dir, obj_ft_dir, obj_det_dir, anno_dir, [scan]
        )
        val_loader = DataLoader(
            val_dataset, cfg.TRAIN.BATCH_SIZE, shuffle=False, 
            pin_memory=True, collate_fn=collate_fn, num_workers=0
        )
        print('%s: objs %d' % (scan, len(val_dataset)))
        with jsonlines.open(os.path.join(pred_dir, '%s.jsonl'%scan), 'w', flush=True) as outf:
            for batch in val_loader:
                if beam == 0:
                    pred_caps = cap_model.greedy_inference(batch)
                    outs = [{bid: pred_caps[j].strip()} for j, bid in enumerate(batch['names'])]
                else:
                    pred_caps = cap_model.beam_inference(batch, beam_size=beam)
                    outs = [{name: res} for name, res in zip(batch['names'], pred_caps)]
            
                for item in outs:
                    outf.write(item)
        
        print('\t%s time: %.2fmin' % (scan, (time.time()-st_time)/60))
    

def validate(cfg, beam, num_workers, scan_sidx, scan_eidx):
    scans = [x.strip() for x in open(os.path.join(connectivity_dir, 'scans.txt'))]
    scans.sort()
    scans = scans[scan_sidx: scan_eidx]
    print('#scene ids: %d' % (len(scans)))

    mp.set_start_method('spawn')

    num_workers = min(num_workers, len(scans))
    num_data_per_worker = len(scans) // num_workers

    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = sidx + num_data_per_worker if proc_id < num_workers - 1 else None
        process = mp.Process(
            target=proc_generate_captions,
            args=(proc_id, scans[sidx: eidx], beam, cfg)
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--resume_epoch', type=int, required=True)
    parser.add_argument('--beam', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--scan_sidx', type=int, default=0)
    parser.add_argument('--scan_eidx', type=int, default=None)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.RESUME_FILE = os.path.join(cfg.OUTPUT_DIR, 'ckpts', 'epoch_%d.pth'%args.resume_epoch)
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.freeze()

    validate(
        cfg, 
        beam=args.beam, 
        num_workers=args.num_workers,
        scan_sidx=args.scan_sidx, scan_eidx=args.scan_eidx,
    )


if __name__ == '__main__':
    main()
        
