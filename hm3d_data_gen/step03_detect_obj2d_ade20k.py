#!/usr/bin/env python3

''' Script to extract bboxes from 36 view images (WIDTH=224, HEIGHT=224, VFOV=60) on HM3D dataset.
'''

import os
import sys
import argparse
import glob
import lmdb
import numpy as np
import json
import jsonlines
import time
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

home_dir = os.environ['HOME']
data_dir = '../datasets'

hm3d_dir = os.path.join(data_dir, 'HM3D')
nav_graph_dir = os.path.join(hm3d_dir, 'nav_graphs_v1')
image_lmdb_dir = os.path.join(nav_graph_dir, 'view_images')

def build_model():
    det_code_dir = os.path.join(os.environ['HOME'], 'codes', 'Mask2Former')
    sys.path.append(det_code_dir)

    # import some common detectron2 utilities
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.utils.visualizer import Visualizer, ColorMode

    # import Mask2Former project
    from mask2former import add_maskformer2_config

    det_cfg = get_cfg()
    add_deeplab_config(det_cfg)
    add_maskformer2_config(det_cfg)

    det_cfg.merge_from_file(
        os.path.join(det_code_dir, 'configs', 'ade20k', 'panoptic-segmentation', 'swin',
                    "maskformer2_swin_large_IN21k_384_bs16_160k.yaml"
        ))
    det_cfg.MODEL.WEIGHTS = os.path.join(data_dir, 'pretrained/segmentation/mask2former/panoptic/model_final_e0c58e.pkl')

    det_cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    det_cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    det_cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    det_cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.8

    det_cfg.freeze()

    metadata = MetadataCatalog.get(
        det_cfg.DATASETS.TEST[0] if len(det_cfg.DATASETS.TEST) else "__unused"
    )

    CLASS_NAMES = np.array(metadata.stuff_classes)
    CLASS_COLORS = np.array(metadata.stuff_colors)

    predictor = DefaultPredictor(det_cfg)

    return predictor


def process_features(proc_id, scene_ids, outdir):
    
    print('start proc_id: %d' % proc_id)

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)
    model = build_model()

    for scene_id in tqdm(scene_ids):
        scene_outdir = os.path.join(outdir, scene_id)
        if os.path.exists(scene_outdir):
            continue

        img_lmdb_path = os.path.join(image_lmdb_dir, f'{scene_id}.lmdb')
        img_env = lmdb.open(img_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

        os.makedirs(scene_outdir, exist_ok=True)
        seg_outfile = os.path.join(scene_outdir, 'panoptic_seg.lmdb')
        bbox_outfile = os.path.join(scene_outdir, 'view_bboxes.jsonl')
        seg_lmdb_env = lmdb.open(seg_outfile, map_size=int(1e12))
        bbox_outf = jsonlines.open(bbox_outfile, 'w', flush=True)

        img_txn = img_env.begin() 
        for vp, value in img_txn.cursor():
            vp = vp.decode('ascii')
            scanvp = '%s_%s'%(scene_id, vp)

            images_flt = np.frombuffer(value, dtype=np.uint8)
            images_flt = cv2.imdecode(images_flt, cv2.IMREAD_COLOR)
            images = images_flt.reshape(36, 224, 224, 3)

            bboxes, segs = [], []
            for im in images:
                panoptic_seg, segments_info = model(im)['panoptic_seg']
            
                panoptic_seg = panoptic_seg.cpu().numpy().astype(np.uint8)
                segs.append(panoptic_seg)

                im_bboxes = []
                for i, item in enumerate(segments_info):
                    mask = panoptic_seg == item['id']
                    x, y = np.nonzero(mask)
                    xyxy = (np.min(y).item(), np.min(x).item(), np.max(y).item()+1, np.max(x).item()+1)
                    im_bboxes.append({
                        'id': item['id'],
                        'score': item['score'],
                        'class': item['category_id'],
                        'xyxy': xyxy,
                    })
                bboxes.append(im_bboxes)
            
            segs = np.concatenate(segs, 0)
            _, segs = cv2.imencode('.png', segs, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            seg_txn = seg_lmdb_env.begin(write=True)
            seg_txn.put(scanvp.encode('ascii'), segs)
            seg_txn.commit()

            bbox_outf.write({'scanvp': scanvp, 'bboxes': bboxes})

        img_env.close() 
        seg_lmdb_env.close()
        bbox_outf.close()
        

def extract_view_image_bboxes(args):
    mp.set_start_method('spawn')

    scene_ids = os.listdir(os.path.join(hm3d_dir, 'train')) + \
                os.listdir(os.path.join(hm3d_dir, 'val'))
    scene_ids.sort(key=lambda x: int(x.split('-')[0]))
    scene_ids = scene_ids[args.start: args.end]

    num_workers = min(args.num_workers, len(scene_ids))
    num_data_per_worker = len(scene_ids) // num_workers

    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=process_features,
            args=(proc_id, scene_ids[sidx: eidx], args.output_dir)
        )
        process.start()
        processes.append(process)
    
    for process in processes:
        process.join()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()

    extract_view_image_bboxes(args)

