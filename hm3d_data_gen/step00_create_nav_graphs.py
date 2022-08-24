"""
Convert HM3D into discrete navigation graphs
"""

import os
import argparse
import numpy as np
import json
import math
import networkx as nx

import habitat_sim
import MatterSim

data_dir = '../datasets'
hm3d_dir = os.path.join(data_dir, 'HM3D')


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
            actuation=habitat_sim.agent.ActuationSpec(amount=5, constraint=None)
        ),
        'look_down': habitat_sim.agent.ActionSpec(
            name='look_down', 
            actuation=habitat_sim.agent.ActuationSpec(amount=5, constraint=None)
        ),
    }
    agent_cfg.height = 1.5 # height of the agent
    agent_cfg.radius = 0.1 # size of the agent

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

    return cfg

def calc_gdist_matrix(sim, cands):
    gdists = np.zeros((len(cands), len(cands))).astype(np.float32)
    path = habitat_sim.ShortestPath()
    for i, icand in enumerate(cands):
        path.requested_start = icand
        for j in range(i+1, len(cands)):
            path.requested_end = cands[j]
            sim.pathfinder.find_path(path)
            gdists[i, j] = gdists[j, i] = path.geodesic_distance
    return gdists

def sample_nav_graph(sim, euc_thresh=2, geo_thresh=3, depth_thresh=1):
    agent = sim.initialize_agent(0)

    nav_points = []
    for t in range(20000):
        nav_points.append(sim.pathfinder.get_random_navigable_point())
    nav_points = np.array(nav_points)

    cands = [nav_points[0]]
    rms = np.zeros((len(nav_points), ), dtype=np.bool)
    rms[0] = True
    while np.sum(rms == 0) > 0:
        c = cands[-1]
        dists = []
        for j in range(len(nav_points)):
            if rms[j]:
                continue
            d = np.sqrt(np.sum((nav_points[j] - c)**2))
            if d < euc_thresh:
                rms[j] = True
            else:
                path = habitat_sim.ShortestPath()
                path.requested_start = c
                path.requested_end = nav_points[j]
                found_path = sim.pathfinder.find_path(path)
                dists.append((j, d, path.geodesic_distance))
        if len(dists) > 0:
            min_idx = np.argmin([x[2] for x in dists])
            rms[dists[min_idx][0]] = True
            if np.isinf(dists[min_idx][-1]):
                remained_idxs = [k for k, rm in enumerate(rms) if not rm]
                if len(remained_idxs) > 0:
                    cands.append(nav_points[np.random.choice(remained_idxs)])
            else:
                cands.append(nav_points[dists[min_idx][0]])
        
    num_cands = len(cands)
    gdists = calc_gdist_matrix(sim, cands)

    edges = []
    for i in range(num_cands):
        for j in range(i+1, num_cands):
            if gdists[i, j] < geo_thresh:
                # visible
                agent_state = habitat_sim.AgentState()
                agent_state.position = cands[i]
                rel_heading = np.pi/2 - np.arctan2( -cands[j][2] + cands[i][2], cands[j][0] - cands[i][0])
                heading = 2 * np.pi - rel_heading
                rel_elevation = np.arctan2(
                    cands[j][1] - cands[i][1], 
                    np.sqrt((cands[i][0]-cands[j][0])**2 + (cands[i][2]-cands[j][2])**2)
                )
                agent_state.rotation = [0, np.sin(heading / 2), 0, np.cos(heading / 2)]
                agent.set_state(agent_state)
                obs = sim.get_sensor_observations()
                n_elevation = np.round(np.rad2deg(np.abs(rel_elevation)) / 5).astype(np.int64)
                for t in range(n_elevation):
                    if rel_elevation > 0:
                        obs = sim.step('look_up')
                    else:
                        obs = sim.step('look_down')
                if np.mean(obs['DEPTH'][100:120, 100:120]) > depth_thresh:
                    edges.append((i, j))

    # clean graph: remove some isolated parts
    G = nx.Graph()
    positions =  {}
    for i, cand in enumerate(cands):
        positions[i] = cand
    for s, e in edges:
        G.add_edge(s, e, weight=1)
    nx.set_node_attributes(G, values=positions, name='position')
    shortest_paths = dict(nx.all_pairs_dijkstra_path(G))

    cleaned_point_idxs = set()
    for i, v in shortest_paths.items():
        if len(v) > 4:
            cleaned_point_idxs.add(i)
    cleaned_point_idxs = list(cleaned_point_idxs)
    cleaned_point_idxs.sort()
    
    idx_map = {oldid: newid for newid, oldid in enumerate(cleaned_point_idxs)}

    # generate a candidate connectivity file
    outs = []
    for i in cleaned_point_idxs:
        cand = cands[i].tolist()
        unobstructed = [False for _ in range(len(cleaned_point_idxs))]
        for start, end in edges:
            if start == i:
                unobstructed[idx_map[end]] = True
            if end == i:
                unobstructed[idx_map[start]] = True
        outs.append({
            'image_id': '%06d'%i,
            'pose': [0, 0, 0, cand[0], 0, 0, 0, -cand[2], 0, 0, 0, cand[1], 0, 0, 0, 1],
            'included': True,
            'unobstructed': unobstructed,
            'height': 1.5,
        })
    return outs

def create_nav_graphs(output_dir):
    np.random.seed(0)

    scene_ids = os.listdir(os.path.join(hm3d_dir, 'train')) + os.listdir(os.path.join(hm3d_dir, 'val'))
    scene_ids.sort(key=lambda x: int(x.split('-')[0]))

    os.makedirs(output_dir, exist_ok=True)
    scan_file = os.path.join(output_dir, 'scans.txt')

    for scene_id in scene_ids:
        print(scene_id)

        sim_cfg = build_simulator_config(scene_id)
        sim = habitat_sim.Simulator(sim_cfg)
        sim.seed(0)

        outs = sample_nav_graph(sim)
        sim.close()
        print('\t#nodes', len(outs))

        with open(scan_file, 'a') as outf:
            print(scene_id, file=outf)
        with open(os.path.join(output_dir, '%s_connectivity.json'%scene_id), 'w') as outf:
            json.dump(outs, outf)


def create_scanvp_candidates(connectivity_dir, outfile):
    # Build simulator
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(640, 480)
    sim.setCameraVFOV(60)
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()

    def _loc_distance(loc):
        return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)

    # Run simulator
    outs = {}
    scans = [x.strip() for x in open(os.path.join(connectivity_dir, 'scans.txt')).readlines()]
    print('#scans: %d' % len(scans))

    for k, scan in enumerate(scans):
        items = json.load(open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)))
        for item in items:
            if not item['included']:
                continue
            viewpoint = item['image_id']
            key = '%s_%s'%(scan, viewpoint)
            outs[key] = {}
            for ix in range(36):
                if ix == 0:
                    sim.newEpisode([scan], [viewpoint], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    sim.makeAction([0], [1], [1])
                else:
                    sim.makeAction([0], [1], [0])
                state = sim.getState()[0]
                assert state.viewIndex == ix
                for cand in state.navigableLocations[1:]:
                    outs[key].setdefault(cand.viewpointId, [])
                    dist = _loc_distance(cand)
                    outs[key][cand.viewpointId].append((ix, dist, cand.rel_heading, cand.rel_elevation))
            res = {}
            for vp, vp_dists in outs[key].items():
                minidx = np.argmin([x[1] for x in vp_dists])
                res[vp] = vp_dists[minidx]
            outs[key] = res

    print('finished', len(outs))
    with open(outfile, 'w') as outf:
        json.dump(outs, outf)

    avg_ncands = []
    for k, v in outs.items():
        avg_ncands.append(len(v))
    print(
        np.min(avg_ncands), np.percentile(avg_ncands, 5),
        np.percentile(avg_ncands, 10), np.mean(avg_ncands), 
        np.percentile(avg_ncands, 90),
        np.percentile(avg_ncands, 95), np.max(avg_ncands)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir')
    args = parser.parse_args()

    create_nav_graphs(args.output_dir) 

    outfile = os.path.join(
        os.path.dirname(args.output_dir), 
        'annotations', 'scanvp_candview_relangles.json'
    )
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    create_scanvp_candidates(args.output_dir, outfile)

