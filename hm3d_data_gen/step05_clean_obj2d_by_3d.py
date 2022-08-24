import os
import sys
import numpy as np
import json
import collections
import glob
import time
import copy
import jsonlines
import lmdb
import cv2
import argparse
from tqdm import tqdm


data_dir = '../datasets'
hm3d_dir = os.path.join(data_dir, 'HM3D')
nav_graph_dir = os.path.join(hm3d_dir, 'nav_graphs_v1')
anno_dir = os.path.join(nav_graph_dir, 'annotations')
conn_dir = os.path.join(nav_graph_dir, 'connectivity')

det_resdir = os.path.join(nav_graph_dir, 'features/obj2d_ade20k')
det_3d_resdir = os.path.join(nav_graph_dir, 'features', 'obj2d_ade20k_pseudo3d')


HEIGHT = WIDTH = 224
HFOV = 60
FOCAL_LEN = WIDTH/2 / np.tan(np.deg2rad(HFOV/2))


def compute_center_direction(bbox, view_id, im_width, im_height, focal_len):
    view_heading = (view_id % 12) * np.deg2rad(30)
    view_elevation = (view_id // 12 - 1) * np.deg2rad(30)
    
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    obj_heading = np.arctan2((x1+x2)/2 - im_width/2, focal_len)
    obj_elevation = np.arctan2((y1+y2)/2 - im_height/2, focal_len)
    center = [obj_heading + view_heading, -obj_elevation + view_elevation]
    
    return center

def get_bbox_size(xyxy):
    return (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])

def merge_scanvp_bboxes_by_3d(scan):
    outfile = os.path.join(det_resdir, scan, 'view_bboxes_merged_by_3d.jsonl')
    if os.path.exists(outfile):
        return

    bboxes_3d = json.load(open(os.path.join(det_3d_resdir, scan, '3d_bboxes.json')))
    scanvp_bboxes_2d = {}
    with jsonlines.open(os.path.join(det_3d_resdir, scan, '2d_bboxes.jsonl'), 'r') as f:
        for item in f:
            scanvp_bboxes_2d[item['scanvp']] = item['bboxes']

    outf_all = jsonlines.open(os.path.join(det_resdir, scan, 'view_bboxes_merged_by_3d_all.jsonl'), 'w')

    with jsonlines.open(outfile, 'w') as outf:
        for scanvp, bboxes_2d in scanvp_bboxes_2d.items():
            obj_3did_to_2d = {}
            for viewid, bboxes in enumerate(bboxes_2d):
                for bbox in bboxes:
                    box_3d_id = str(bbox['3d_bbox_id'])
                    obj_3did_to_2d.setdefault(box_3d_id, [])
                    assert bboxes_3d[box_3d_id]['class'] == bbox['class']
                    obj_3did_to_2d[box_3d_id].append({
                        'view_id': viewid,
                        'obj_id': box_3d_id,
                        'xyxy': bbox['xyxy'],
                        'depth': bbox['depth'],
                        'score': bbox['score'],
                        'center': compute_center_direction(bbox['xyxy'], viewid, HEIGHT, WIDTH, FOCAL_LEN),
                        'inst_id': bbox['inst_id'],
                        'class': bbox['class'],
                        'name': bbox['name'],
                        '3d_center': bboxes_3d[box_3d_id]['center'],
                        '3d_size': bboxes_3d[box_3d_id]['size'],
                    })
            obj_3did_to_2d_max = []
            for obj_3did, value in obj_3did_to_2d.items():
                bbox_2d_sizes = [get_bbox_size(v['xyxy']) for v in value]
                max_idx = np.argmax(bbox_2d_sizes)
                obj_3did_to_2d_max.append(value[max_idx])

            outf.write({'scanvp': scanvp, 'bboxes': obj_3did_to_2d_max})
            outf_all.write({'scanvp': scanvp, 'bboxes': obj_3did_to_2d})

    outf_all.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()

    scene_ids = os.listdir(os.path.join(hm3d_dir, 'train')) + \
                os.listdir(os.path.join(hm3d_dir, 'val'))
    scene_ids.sort(key=lambda x: int(x.split('-')[0]))
    scene_ids = scene_ids[args.start: args.end]

    for scene_id in tqdm(scene_ids):
        merge_scanvp_bboxes_by_3d(scene_id)

if __name__ == '__main__':
    main()