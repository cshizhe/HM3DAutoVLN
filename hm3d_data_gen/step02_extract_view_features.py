#!/usr/bin/env python3

''' Script to precompute HM3D view image features (WIDTH = 224, HEIGHT = 224, VFOV = 60).
'''

import os
import argparse
import lmdb
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

import torch
import torch.multiprocessing as mp

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

data_dir = '../datasets'

hm3d_dir = os.path.join(data_dir, 'HM3D')
nav_graph_dir = os.path.join(hm3d_dir, 'nav_graphs_v1')
image_lmdb_dir = os.path.join(nav_graph_dir, 'view_images')


def build_timm_feature_extractor(model_name, gpu_id=0):
    device = torch.device('cuda:%d'%gpu_id if torch.cuda.is_available() else 'cpu')

    model = timm.create_model(model_name, pretrained=True).to(device)
    model.eval()

    config = resolve_data_config({}, model=model)
    # {'input_size': (3, 224, 224),
    # 'interpolation': 'bicubic',
    # 'mean': (0.5, 0.5, 0.5),
    # 'std': (0.5, 0.5, 0.5),
    # 'crop_pct': 0.9}
    config['crop_pct'] = 1
    img_transforms = create_transform(**config)

    return model, img_transforms, device


def process_features(proc_id, scene_ids, args):
    print('start proc_id: %d' % proc_id)

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)

    model, img_transforms, device = build_timm_feature_extractor(args.model_name)
    
    if proc_id == 0:
        print('model', args.model_name)
        print(model)
        print('\nimage transforms')
        print(img_transforms)

    for scene_id in tqdm(scene_ids):
        lmdb_path = os.path.join(image_lmdb_dir, f'{scene_id}.lmdb')
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

        txn = env.begin() 
        for key, value in txn.cursor():
            scanvp = '%s_%s' % (scene_id, key.decode('ascii'))
            output_file = os.path.join(args.output_dir, '%s.npy'%scanvp)
            if os.path.exists(output_file):
                continue

            images_flt = np.frombuffer(value, dtype=np.uint8)
            images_flt = cv2.imdecode(images_flt, cv2.IMREAD_COLOR)
            images = images_flt.reshape(36, 224, 224, 3)
            images = [Image.fromarray(image) for image in images]
            images = torch.stack([img_transforms(image).to(device) for image in images], 0)

            fts = model.forward_features(images)
            logits = model.head(fts)
            fts = torch.cat([fts, logits], dim=1)
            fts = fts.data.cpu().numpy()
            
            np.save(output_file, fts)

        env.close()


def build_feature_file(args):
    mp.set_start_method('spawn')

    os.makedirs(args.output_dir, exist_ok=True)

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
            args=(proc_id, scene_ids[sidx: eidx], args)
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
        fts = np.load(os.path.join(args.output_dir, npy_file))

        txn = env.begin(write=True)
        txn.put(scanvp.encode('ascii'), msgpack.packb(fts))
        txn.commit()

    env.close()

    for npy_file in npy_files:
        os.remove(os.path.join(args.output_dir, npy_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', choices=[
        'vit_base_patch16_224',
    ], default='vit_base_patch16_224')
    parser.add_argument('--output_dir')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()

    build_feature_file(args)
