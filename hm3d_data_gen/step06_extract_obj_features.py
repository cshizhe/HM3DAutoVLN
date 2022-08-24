#!/usr/bin/env python3

''' Script to precompute REVERIE groundtruth object features.
'''

import os
import sys
import argparse
import numpy as np
import jsonlines
import lmdb
import cv2
from PIL import Image
from tqdm import tqdm

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

data_dir = '../datasets'

hm3d_dir = os.path.join(data_dir, 'HM3D')
nav_graph_dir = os.path.join(hm3d_dir, 'nav_graphs_v1')
image_lmdb_dir = os.path.join(nav_graph_dir, 'view_images')
det_resdir = os.path.join(nav_graph_dir, 'features', 'obj2d_ade20k')


def build_timm_feature_extractor(model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = timm.create_model(model_name, pretrained=True).to(device)
    model.eval()

    config = resolve_data_config({}, model=model)
    config['crop_pct'] = 1
    img_transforms = create_transform(**config)

    return model, img_transforms, device

def process_features(proc_id, scans, args):
    print('start proc_id: %d' % proc_id)

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)

    model, img_transforms, device = build_timm_feature_extractor(args.model_name)

    if proc_id == 0:
        print(args.model_name)
        print(model)
        print('\nimage transforms')
        print(img_transforms)

    for scan in tqdm(scans):
        scan_bboxes = []
        with jsonlines.open(os.path.join(det_resdir, scan, 'view_bboxes_merged_by_3d.jsonl'), 'r') as f:
            for item in f:
                scan_bboxes.append(item)

        for item in scan_bboxes:
            scanvp = item['scanvp']
            scan, vp = scanvp.split('_')

            output_file = os.path.join(args.output_dir, '%s.npy'%scanvp)
            if os.path.exists(output_file):
                continue

            obj_bboxes = item['bboxes']

            # load images
            lmdb_path = os.path.join(image_lmdb_dir, f'{scan}.lmdb')
            with lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False) as env:
                with env.begin() as txn:
                    images_flt = txn.get(vp.encode('ascii'))
                    images_flt = np.frombuffer(images_flt, dtype=np.uint8)
                    images_flt = cv2.imdecode(images_flt, cv2.IMREAD_COLOR)
                    images = images_flt.reshape(36, 224, 224, 3)

            obj_images = []
            for value in obj_bboxes:
                image = images[value['view_id']]
                box = np.round(value['xyxy']).astype(np.long)
                obj_image = image[box[1]: box[3], box[0]: box[2]]
                obj_images.append(Image.fromarray(obj_image))

            view_ids = [v['view_id'] for v in obj_bboxes]
            # obj_ids = [str(objid) for objid in range(len(obj_bboxes))] # fake id
            obj_ids = [v['obj_id'] for v in obj_bboxes]
            obj_xywhs = [
                [v['xyxy'][0], v['xyxy'][1], v['xyxy'][2]-v['xyxy'][0], v['xyxy'][3]-v['xyxy'][1]] \
                    for v in obj_bboxes
            ]
            obj_centers = [v['center'] for v in obj_bboxes]
            obj_names = [v['name'] for v in obj_bboxes]
            obj_3d_centers = [v['3d_center'] for v in obj_bboxes]
            obj_3d_sizes = [v['3d_size'] for v in obj_bboxes]

            if args.save_images and args.image_dir is not None:
                os.makedirs(args.image_dir, exist_ok=True)
                for k, image in enumerate(obj_images):
                    image.save(os.path.join(args.image_dir, '%s_%s.jpg'%(scanvp, obj_ids[k])))

            if len(obj_images) > 0:
                obj_images = torch.stack([img_transforms(image).to(device) for image in obj_images], 0)
                fts = []
                for k in range(0, len(obj_images), args.batch_size):
                    b_fts = model.forward_features(obj_images[k: k+args.batch_size])
                    b_logits = model.head(b_fts)
                    b_fts = torch.cat([b_fts, b_logits], 1)
                    b_fts = b_fts.data.cpu().numpy()
                    fts.append(b_fts)
                fts = np.concatenate(fts, 0)
                res = {
                    'scanvp': scanvp, 
                    'obj_ids': obj_ids,
                    'view_ids': view_ids,
                    'obj_names': obj_names,
                    'bboxes': obj_xywhs,
                    'centers': obj_centers,
                    'fts': fts,
                    '3d_centers': obj_3d_centers,
                    '3d_sizes': obj_3d_sizes,
                }
                np.save(output_file, res)
            

def build_feature_file(args):
    mp.set_start_method('spawn')

    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    scans = os.listdir(det_resdir)
    scans.sort()
    scans = scans[args.start: args.end]

    num_workers = min(args.num_workers, len(scans))
    num_data_per_worker = len(scans) // num_workers

    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=process_features,
            args=(proc_id, scans[sidx: eidx], args)
        )
        process.start()
        processes.append(process)
    
    for process in processes:
        process.join()
            
    # merge all npy files into a lmdb file
    npy_files = [x for x in os.listdir(args.output_dir) if x.endswith('.npy')]
    npy_files.sort()

    env = lmdb.open(args.output_dir, map_size=int(1e12))
    for npy_file in npy_files:
        scanvp = os.path.splitext(npy_file)[0]
        tmp = np.load(os.path.join(args.output_dir, npy_file), allow_pickle=True).item()

        txn = env.begin(write=True)
        txn.put(scanvp.encode('ascii'), msgpack.packb(tmp))
        txn.commit()

    env.close()

    for npy_file in npy_files:
        os.remove(os.path.join(args.output_dir, npy_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', choices=[
        'vit_base_patch16_224'
    ], default='vit_base_patch16_224')
    parser.add_argument('--save_images', action='store_true', default=False)
    parser.add_argument('--image_dir', default=None)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--output_dir')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    build_feature_file(args)

