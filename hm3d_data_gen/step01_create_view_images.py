'''
Extract 36 view images for nav_graphs of HM3D
'''

import os
import argparse
import numpy as np
import cv2
import json
import lmdb
import time
import multiprocessing as mp

import habitat_sim

data_dir = '../datasets'
hm3d_dir = os.path.join(data_dir, 'HM3D')

connectivity_dir = os.path.join(hm3d_dir, 'nav_graphs_v1', 'connectivity')
output_dir = os.path.join(hm3d_dir, 'nav_graphs_v1', 'view_images')


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


def extract_scene_view_images(scene_id, lmdb_path, connectivity_path):
    sim_cfg = build_simulator_config(scene_id)
    sim = habitat_sim.Simulator(sim_cfg)
    sim.seed(0)
    agent = sim.initialize_agent(0)

    data = json.load(open(connectivity_path))

    os.makedirs(lmdb_path, exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=int(1e12))

    st_time = time.time()
    for item in data:
        x, z, y = item['pose'][3], -item['pose'][7], item['pose'][11] - item['height']
        agent_state = habitat_sim.AgentState()
        agent_state.position = [x, y, z] # world space
        agent.set_state(agent_state)
        obs = sim.get_sensor_observations()
        
        images = []
        # down
        obs = sim.step('look_down')
        for i in range(12):
            images.append(obs['RGB'][..., :-1])
            obs = sim.step('turn_right')
        # middle
        obs = sim.step('look_up')
        for i in range(12):
            images.append(obs['RGB'][..., :-1])
            obs = sim.step('turn_right')
        # up
        obs = sim.step('look_up')
        for i in range(12):
            images.append(obs['RGB'][..., :-1])
            obs = sim.step('turn_right')
        
        images = np.concatenate(images, 0)

        # compress images
        _, image_bytes = cv2.imencode('.png', images, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        
        txn = env.begin(write=True)
        item_key = item['image_id'].encode('ascii')
        txn.put(item_key, image_bytes)
        txn.commit()
            
    env.close()
    sim.close()
    print('%s: vps %d, time %.2fmin' %(scene_id, len(data), (time.time()-st_time)/60.))
        
def extract_view_images(scene_ids, connectivity_dir, output_dir, out_queue):
    for scene_id in scene_ids:
        connectivity_path = os.path.join(connectivity_dir, '%s_connectivity.json'%scene_id)
        lmdb_path = os.path.join(output_dir, '%s.lmdb'%scene_id)
        if os.path.exists(lmdb_path):
            out_queue.put(scene_id)
        else:
            extract_scene_view_images(scene_id, lmdb_path, connectivity_path)
            out_queue.put(scene_id)

def main(args):
    np.random.seed(0)

    scene_ids = os.listdir(os.path.join(hm3d_dir, 'train')) + os.listdir(os.path.join(hm3d_dir, 'val'))
    scene_ids.sort(key=lambda x: int(x.split('-')[0]))
    scene_ids = scene_ids[args.start: args.end]

    num_workers = min(len(scene_ids), args.num_workers)
    num_jobs_per_worker = len(scene_ids) // num_workers

    procs = []
    out_queue = mp.Queue()
    for i in range(num_workers):
        sidx = i * num_jobs_per_worker
        eidx = sidx + num_jobs_per_worker if i < num_workers - 1 else None
        procs.append(mp.Process(
            target=extract_view_images,
            args=(scene_ids[sidx: eidx], connectivity_dir, output_dir, out_queue)
        ))
        procs[-1].start()

    for i in range(len(scene_ids)):
        scene_id = out_queue.get()
        print('%04d/%04d: %s' %(i, len(scene_ids), scene_id))

    for proc in procs:
        proc.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()

    main(args)