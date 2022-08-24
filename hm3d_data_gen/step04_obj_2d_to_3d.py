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

import torch
import habitat_sim

sys.path.append('../../pcd_utils')
import rotation_utils as ru
import depth_utils as du

data_dir = '../datasets'

hm3d_dir = os.path.join(data_dir, 'HM3D')
nav_graph_dir = os.path.join(hm3d_dir, 'nav_graphs_v1')
anno_dir = os.path.join(nav_graph_dir, 'annotations')
conn_dir = os.path.join(nav_graph_dir, 'connectivity')
det_resdir = os.path.join(nav_graph_dir, 'features/obj2d_ade20k')

output_dir = os.path.join(nav_graph_dir, 'features', 'obj2d_ade20k_pseudo3d')

stuff_classes = json.load(open(os.path.join(anno_dir, 'ade20k_stuff_classes.json'), 'r'))
stuff_colors = json.load(open(os.path.join(anno_dir, 'ade20k_stuff_colors.json'), 'r'))
indoor_classes = json.load(open(os.path.join(anno_dir, 'ade20k_indoor_classes.json'), 'r'))

stuff_indoor_idxs = [stuff_classes.index(w) for w in indoor_classes]
indoor_name2cid = {w: i for i, w in enumerate(indoor_classes)}
indoor_colors = np.array([stuff_colors[i] for i in stuff_indoor_idxs])

stuff_indoor_idxs = set(stuff_indoor_idxs)
num_classes = len(indoor_classes)

MAP_RESOLUTION = 0.1
OBJ_MIN_DEPTH = 4
MIN_POINTS_PER_VOXEL = 2


def build_simulator_config(scene_id):
    # make configurations
    sim_cfg = habitat_sim.SimulatorConfiguration()
    if int(scene_id.split('-')[0]) < 800:
        split = 'train'
    else:
        split = 'val'
    sim_cfg.scene_id = os.path.join(hm3d_dir, split, scene_id, scene_id.split('-')[-1]+'.basis.glb')

    sim_cfg.default_agent_id = 0
    sim_cfg.enable_physics = False
    sim_cfg.allow_sliding = True
    sim_cfg.gpu_device_id = 0
    sim_cfg.random_seed = 0

    agent_cfg = habitat_sim.agent.AgentConfiguration()

    resolution = [224, 224] # (image_height, image_width)
    position = [0, 1.5, 0] # (_, sensor_height, _)

    sensor_specs = []

    sensor_cfg = habitat_sim.SensorSpec()
    sensor_cfg.uuid = f'RGB'
    sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
    sensor_cfg.resolution = resolution
    sensor_cfg.position = position
    sensor_cfg.orientation = [0, 0, 0]
    sensor_cfg.parameters['hfov'] = '60' # MapStringString{far: 1000, hfov: 90, near: 0.01, ortho_scale: .1}
    sensor_specs.append(sensor_cfg)

    sensor_cfg = habitat_sim.SensorSpec()
    sensor_cfg.uuid = f'DEPTH'
    sensor_cfg.sensor_type = habitat_sim.SensorType.DEPTH
    sensor_cfg.resolution = resolution
    sensor_cfg.position = position
    sensor_cfg.orientation = [0, 0, 0]
    sensor_cfg.parameters['hfov'] = '60' # MapStringString{far: 1000, hfov: 90, near: 0.01, ortho_scale: .1}
    sensor_specs.append(sensor_cfg)

    agent_cfg.sensor_specifications = sensor_specs

    agent_cfg.action_space = {
        'move_forward': habitat_sim.agent.ActionSpec(
            name='move_forward', 
            actuation=habitat_sim.agent.ActuationSpec(amount=0.25, constraint=None)
        ),
        'turn_left': habitat_sim.agent.ActionSpec(
            name='turn_left', 
            actuation=habitat_sim.agent.ActuationSpec(amount=30, constraint=None)
        ),
        'turn_right': habitat_sim.agent.ActionSpec(
            name='turn_right', 
            actuation=habitat_sim.agent.ActuationSpec(amount=30, constraint=None)
        ),
        'look_up': habitat_sim.agent.ActionSpec(
            name='look_up', 
            actuation=habitat_sim.agent.ActuationSpec(amount=30, constraint=None)
        ),
        'look_down': habitat_sim.agent.ActionSpec(
            name='look_down', 
            actuation=habitat_sim.agent.ActuationSpec(amount=30, constraint=None)
        ),
    }
    agent_cfg.height = 1.5 # height of the agent
    agent_cfg.radius = 0.1 # size of the agent

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    return cfg


def find_neighboring_voxels(voxel, s=1):
    x, y, z = voxel
    neighbors = []
    for ix in range(x-s, x+s+1):
        for iy in range(y-s, y+s+1):
            for iz in range(z-s, z+s+1):
                neighbors.append((ix, iy, iz))
    return neighbors

def enlarge_obj_voxels(obj_voxels, cid, grids):
    enlarged = set(obj_voxels)
    todo = list(enlarged)
    while len(todo) > 0:
        neighbors = [x for x in find_neighboring_voxels(todo[0]) if x in grids and x not in enlarged]
        for v in neighbors:
            if v not in enlarged:
                if np.argmax(grids[v]['sem']) == cid:
                    todo.append(v)
                    enlarged.add(v)
        todo = todo[1:]
    return enlarged

def calc_voxel_overlap(a, b):
    intersect = a.intersection(b)
    return len(intersect) / min(len(a), len(b))

def extract_3d_bboxes_per_scene(scan, outdir):
    st_time = time.time()

    cfg = build_simulator_config(scan)
    sim = habitat_sim.Simulator(cfg)
    sim.seed(0)
    agent = sim.initialize_agent(0)

    # load navigation points
    nav_vps, nav_points = [], []
    d = json.load(open(os.path.join(conn_dir, '%s_connectivity.json'%scan)))
    for x in d:
        loc = [x['pose'][3], x['pose'][7], x['pose'][11]]
        nav_vps.append(x['image_id'])
        nav_points.append([loc[0], loc[2] - x['height'], -loc[1]])
    nav_points = np.array(nav_points) # (x: horizon, y: height, z: vertical)
    print('%s: vps %d' % (scan, len(nav_points)))

    screen_h, screen_w, fov = 224, 224, 60
    agent_height = 1.5
    camera_matrix = du.get_camera_matrix(screen_w, screen_h, fov)

    # load detection results
    bboxes = {}
    n = 0
    with jsonlines.open(os.path.join(det_resdir, scan, 'view_bboxes.jsonl'), 'r') as f:
        for x in f:
            bboxes[x['scanvp']] = []
            for view in x['bboxes']:
                bboxes[x['scanvp']].append([])
                for v in view:
                    if v['class'] in stuff_indoor_idxs:
                        v['class'] = indoor_name2cid[stuff_classes[v['class']]]
                        bboxes[x['scanvp']][-1].append(v)
                        n += 1
    print('\tbboxes: %d' % (n))

    lmdb_path = os.path.join(det_resdir, scan, 'panoptic_seg.lmdb')
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    txn = env.begin() 
    segs = {}
    for key, value in txn.cursor():
        images_flt = np.frombuffer(value, dtype=np.uint8)
        images_flt = cv2.imdecode(images_flt, cv2.IMREAD_GRAYSCALE)
        segs[key.decode('ascii')] = images_flt.reshape(36, 224, 224)
    env.close()

    # extract multiple view images for instance segmentation
    grids = {}
    for nav_vp, nav_point in zip(nav_vps, nav_points):
        scanvp = '%s_%s'%(scan, nav_vp)
        
        agent_state = habitat_sim.AgentState()
        agent_state.position = nav_point # world space
        agent_state.rotation = [0, 0, 0, 1]
        agent.set_state(agent_state)

        obs = sim.step('look_down')
        for e, elevation in enumerate([-30, 0, 30]):
            for i in range(12):
                viewid = e * 12 + i
                
                heading = -i * np.pi / 6
                obs = sim.get_sensor_observations()
                agent_state = agent.get_state()

                depth = torch.from_numpy(obs['DEPTH'])
                habitat_xyz = agent_state.position
                agent_loc = [habitat_xyz[0], -habitat_xyz[2], habitat_xyz[1]]

                point_cloud = du.get_point_cloud_from_z_t(
                    depth.unsqueeze(0), camera_matrix, 'cpu'
                ) 
                agent_view = du.transform_camera_view_t(point_cloud, agent_height + agent_loc[2], elevation, 'cpu')
                shift_loc = [agent_loc[0], agent_loc[1], heading]
                agent_view_centered = du.transform_pose_t(agent_view, shift_loc, 'cpu').squeeze(0).data.numpy()
                
                # downsampling group
                view_points = np.round(agent_view_centered / MAP_RESOLUTION).astype(np.long)

                mask = (obs['DEPTH'] != 0) & (obs['DEPTH'] < 6)
                                        
                for kobj, bbox in enumerate(bboxes[scanvp][viewid]):
                    inst_mask = mask & (segs[scanvp][viewid] == bbox['id'])
                    if np.sum(inst_mask) > 0:
                        bbox_view_points = view_points[inst_mask]
                            
                        # keep distance of obj
                        bbox['depth'] = obs['DEPTH'][inst_mask].mean()
                        key_inds = [tuple(k) for k in np.unique(bbox_view_points, axis=0)]
                        bbox['voxels'] = key_inds

                        if bbox['depth'] < OBJ_MIN_DEPTH:
                            for k in key_inds:
                                grids.setdefault(k, {
                                    'num': 0,
                                    'sem': np.zeros(num_classes, ).astype(np.float32),
                                })
                                grids[k]['num'] += 1
                                grids[k]['sem'][bbox['class']] += bbox['score']

                obs = sim.step('turn_right')

            obs = sim.step('look_up')

    sim.close()

    clean_grids = {}
    for k, v in grids.items():
        if v['num'] >= MIN_POINTS_PER_VOXEL:
            clean_grids[k] = v
    grids = clean_grids
    print('\ttotal voxels: %d' % (len(grids)))

    # map 2d bbox to 3d enlarged bbox
    for scanvp, value in bboxes.items():
        for viewid, view_value in enumerate(value):
            for objid, bbox in enumerate(view_value):
                if 'depth' in bbox and bbox['depth'] < OBJ_MIN_DEPTH:    
                    p = []
                    for obj_voxel in bbox['voxels']:
                        if obj_voxel in grids:
                            p.append(grids[obj_voxel]['sem']/grids[obj_voxel]['num'])
                    if len(p) > 0:
                        p = np.mean(p, 0)
                        new_cid = np.argmax(p)
                        bbox['enlarged_voxels'] = enlarge_obj_voxels(bbox['voxels'], new_cid, grids)
                        bbox['new_score'] = p[new_cid]
                        bbox['new_class'] = new_cid

    # filtered bboxes
    cleaned_bboxes = []
    for scanvp, value in bboxes.items():        
        for viewid, view_value in enumerate(value):
            for objid, bbox in enumerate(view_value):
                if 'enlarged_voxels' not in bbox or bbox['new_score'] < 0.5:
                    continue
                cleaned_bboxes.append({
                    'scanvp': scanvp, 'viewid': viewid, 'objid': objid, 'inst_id': bbox['id'],
                    'xyxy': bbox['xyxy'], 'depth': bbox['depth'],
                    'class': bbox['new_class'],
                    'score': bbox['new_score'],
                    'voxels': bbox['enlarged_voxels'],
                })
    print('\tcleaned 2d bboxes: %d' % (len(cleaned_bboxes)))

    # merge 3d bboxes
    merged_3d_bbox_ids = {}
    used = set()
    for i, vx in enumerate(cleaned_bboxes):
        if i in used: continue
        for j, vy in enumerate(cleaned_bboxes[i:]):
            j += i
            if j in used: continue
            if vx['class'] == vy['class']:
                u = calc_voxel_overlap(vx['voxels'], vy['voxels'])
                if u > 0.5:
                    merged_3d_bbox_ids.setdefault(i, [])
                    merged_3d_bbox_ids[i].append(j)
                    used.add(j)
    print('\tmerged 3d bboxes: %d' % (len(merged_3d_bbox_ids)))

    id2d_to_id3d = {}
    for k, vids in merged_3d_bbox_ids.items():
        for vid in vids:
            id2d_to_id3d[vid] = k

    merged_3d_bboxes_union = {}
    for k, vids in merged_3d_bbox_ids.items():
        merged_3d_bboxes_union[k] = collections.Counter()
        for vid in vids:
            merged_3d_bboxes_union[k].update(cleaned_bboxes[vid]['voxels'])

    for k, v in merged_3d_bboxes_union.items():
        n = np.percentile(list(v.values()), 80)
        merged_3d_bboxes_union[k] = [x for x, c in v.items() if c >= n]

    instid_to_3dbbox = {}
    for k, voxels in merged_3d_bboxes_union.items():
        k_points = np.array(list(voxels))
        cid = cleaned_bboxes[k]['class']        
        instid_to_3dbbox[k] = {
            'center': k_points.mean(0),
            'size': np.percentile(k_points, 95, 0) - np.percentile(k_points, 5, 0) + 1,
            'class': cid,
        }

    # save outputs
    outs_2d_bboxes = {}
    for k, item in enumerate(cleaned_bboxes):
        outs_2d_bboxes.setdefault(item['scanvp'], [[] for _ in range(36)])
        outs_2d_bboxes[item['scanvp']][item['viewid']].append({
            'inst_id': item['inst_id'],
            'xyxy': item['xyxy'],
            'depth': float(item['depth']),
            'class': int(item['class']),
            'name': indoor_classes[item['class']],
            'score': float(item['score']),
            '3d_bbox_id': int(id2d_to_id3d[k])
        })
        
    outs_3d_bboxes = {}
    for k, v in instid_to_3dbbox.items():
        outs_3d_bboxes[k] = {
            'center': (v['center']*MAP_RESOLUTION).tolist(),
            'size': (v['size']*MAP_RESOLUTION).tolist(),
            'class': int(v['class']),
            'name': indoor_classes[v['class']],
        }

    os.makedirs(outdir, exist_ok=True)
    json.dump(outs_3d_bboxes, open(os.path.join(outdir, '3d_bboxes.json'), 'w'))
    with jsonlines.open(os.path.join(outdir, '2d_bboxes.jsonl'), 'w') as outf:
        for scanvp, value in outs_2d_bboxes.items():
            outf.write({
                'scanvp': scanvp,
                'bboxes': value
            })
    print('\tcost time %.2fmin' % ((time.time() - st_time) / 60.))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()

    scene_ids = os.listdir(os.path.join(hm3d_dir, 'train')) + \
                os.listdir(os.path.join(hm3d_dir, 'val'))
    scene_ids.sort(key=lambda x: int(x.split('-')[0]))
    scene_ids = scene_ids[args.start: args.end]

    for scene_id in scene_ids:
        scene_outdir = os.path.join(output_dir, scene_id)
        if not os.path.exists(scene_outdir):
            extract_3d_bboxes_per_scene(scene_id, scene_outdir)

if __name__ == '__main__':
    main()