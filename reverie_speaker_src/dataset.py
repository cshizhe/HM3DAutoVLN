import os
import json
import jsonlines
import collections
import copy
import numpy as np
import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

import torch

def calc_angle_feature(heading, elevation):
    return np.array(
        [np.sin(heading), np.cos(heading), np.sin(elevation), np.cos(elevation)]
        ).astype(np.float32)


class ReveriePanoObject2DCaptionDataset(object):
    def __init__(
        self, view_ft_dir, obj_ft_dir, anno_dir, split,
        view_ft_size=768, obj_ft_size=768, use_ctx_objs=True,
        image_height=224, image_width=224,
        in_memory=False, max_txt_len=100, is_train=True
    ):
        super().__init__()
        self.view_ft_dir = view_ft_dir
        self.obj_ft_dir = obj_ft_dir
        self.anno_dir = anno_dir
        self.split = split
        self.is_train = is_train

        self.view_ft_size = view_ft_size
        self.obj_ft_size = obj_ft_size
        self.use_ctx_objs = use_ctx_objs
        self.image_height = image_height
        self.image_width = image_width

        self.in_memory = in_memory
        if self.in_memory:
            self._view_fts, self._obj_fts = {}, {}
        
        self.captions = []
        if self.split is not None:
            with jsonlines.open(os.path.join(anno_dir, f'gpt2_captions_{split}.jsonl'), 'r') as f:
                for item in f:
                    for k, instr_encoding in enumerate(item['instr_encodings']):
                        newitem = copy.deepcopy(item)
                        del newitem['instr_encodings']
                        newitem['instr_encoding'] = instr_encoding[:max_txt_len]
                        newitem['instr_id'] = k
                        self.captions.append(newitem)
                        if not self.is_train:
                            break
        self.objname_maps = json.load(
            open(os.path.join(anno_dir, 'reverie_to_ade20k_class_map.json'))
        )
        objname_fts = json.load(
            open(os.path.join(anno_dir, 'ade20k_indoor_classes_to_glove42b.json'))
        )
        self.objname_fts = {}
        for k, v in objname_fts.items():
            self.objname_fts[k] = np.array(v, dtype=np.float32)

        self.view_angle_features = []
        for view_id in range(36):
            heading = np.deg2rad((view_id % 12) * 30)
            elevation = np.deg2rad((view_id // 12 - 1) * 30)
            self.view_angle_features.append(
                calc_angle_feature(heading, elevation)
            )
        self.view_angle_features = np.array(self.view_angle_features)
            
    def __len__(self):
        return len(self.captions)

    def get_view_feature(self, scan, vp):
        key = f'{scan}_{vp}'
        if self.in_memory and key in self._view_fts:
            return self._view_fts[key]
        else:
            env = lmdb.open(self.view_ft_dir, readonly=True)
            txn = env.begin()
            view_fts = msgpack.unpackb(txn.get(key.encode('ascii')))
            env.close()

            view_fts = view_fts[:, :self.view_ft_size]
            return view_fts

    def get_obj_feature(self, scan, vp):
        key = f'{scan}_{vp}'
        if self.in_memory and key in self._obj_fts:
            return self._obj_fts[key]
        else:
            # obj_ids, obj_names, fts, view_ids, bboxes (xywh), centers
            env = lmdb.open(self.obj_ft_dir, readonly=True)
            txn = env.begin()
            data = msgpack.unpackb(txn.get(key.encode('ascii')))
            env.close()

            outs = {}
            for i, objid in enumerate(data['obj_ids']):
                heading, elevation = data['centers'][i]
                x, y, w, h = data['bboxes'][i]
                w = w / self.image_width
                h = h / self.image_height
                outs[objid] = {
                    'img_ft': data['fts'][i, :self.obj_ft_size],
                    'name_ft': self.objname_fts[self.objname_maps[data['obj_names'][i]]],
                    'center_ft': calc_angle_feature(heading, elevation),
                    'size_ft': [w, h, w * h],
                    'view_id': data['view_ids'][i]
                }
            if self.in_memory:
                self._obj_fts[key] = outs
            return outs

    def __getitem__(self, idx):
        item = self.captions[idx]
        scan = item['scan']
        if self.is_train:
            vp = np.random.choice(item['pos_vps'])
        else:
            vp = item['pos_vps'][0]
        tgt_objid = item['obj_id']
        scanvpobj = '%s_%s_%s' % (scan, vp, str(tgt_objid))

        obj_data = self.get_obj_feature(scan, vp)
        # image ft, name glove42b, center angle, size
        if self.use_ctx_objs:
            nobjs = len(obj_data)
        else:
            nobjs = 1
        obj_fts = np.zeros((nobjs, self.obj_ft_size + 300 + 4 + 3), dtype=np.float32)
        obj_fts[0, :self.obj_ft_size] = obj_data[tgt_objid]['img_ft']
        obj_fts[0, self.obj_ft_size: self.obj_ft_size+300] = obj_data[tgt_objid]['name_ft']
        obj_fts[0, self.obj_ft_size+300: self.obj_ft_size+304] = obj_data[tgt_objid]['center_ft']
        obj_fts[0, self.obj_ft_size+304: ] = obj_data[tgt_objid]['size_ft']

        if self.use_ctx_objs:
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
            'names': item['id'],
            'ref_caps': item['instructions'],
            'obj_fts': obj_fts,
            'view_fts': view_fts,
            'cap_ids': np.array(item['instr_encoding']),
        }



def collate_fn(data):
    outs = {}
    for key in data[0].keys():
        outs[key] = [x[key] for x in data]
    batch_size = len(data)

    if 'cap_ids' in outs:
        outs['cap_lens'] = torch.LongTensor([len(x) for x in outs['cap_ids']])
        max_txt_len = max(outs['cap_lens'])
        cap_ids = torch.zeros(batch_size, max_txt_len).long()
        for i, cap in enumerate(outs['cap_ids']):
            cap_ids[i][:outs['cap_lens'][i]] = torch.from_numpy(cap)
        outs['cap_ids'] = cap_ids

    if outs['view_fts'][0] is not None:
        outs['view_fts'] = torch.from_numpy(np.stack(outs['view_fts'], 0))
        outs['view_types'] = torch.zeros(batch_size, 36, dtype=torch.long)
        outs['view_types'][:, 0] = 1
    else:
        del outs['view_fts']

    obj_lens = torch.LongTensor([len(x) for x in outs['obj_fts']])
    max_obj_len = max(obj_lens)
    obj_fts = torch.zeros(batch_size, max_obj_len, outs['obj_fts'][0].shape[-1]).float()
    obj_types = torch.zeros(batch_size, max_obj_len).long()
    obj_masks = torch.ones(batch_size, max_obj_len).bool()
    for i in range(batch_size):
        obj_fts[i, :obj_lens[i]] = torch.from_numpy(outs['obj_fts'][i])
        obj_types[i, 0] = 1
        obj_masks[i, :obj_lens[i]] = 0
    outs['obj_fts'] = obj_fts
    outs['obj_types'] = obj_types
    outs['obj_masks'] = obj_masks
    return outs


if __name__ == '__main__':
    import os
    import time
    data_dir = '../datasets'

    dataset = ReveriePanoObject2DCaptionDataset(
        os.path.join(data_dir, 'R2R', 'features', 'view_timm_imagenet_vitb16'),
        os.path.join(data_dir, 'REVERIE', 'features', 'obj_gtmax_timm_imagenet_vitb16'),
        os.path.join(data_dir, 'REVERIE', 'annotations', 'speaker_inputs'),
        'train', view_ft_size=768, obj_ft_size=768, 
        image_height=480, image_width=640,
        in_memory=True, max_txt_len=100, is_train=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True, pin_memory=True,
        collate_fn=collate_fn, num_workers=0
    )

    start_time = time.time()
    cap_lens = []
    for batch in dataloader:
        print(batch)
        break
    #     cap_lens.extend(batch['cap_lens'].numpy().tolist())
    # print(np.min(cap_lens), np.mean(cap_lens), np.percentile(cap_lens, 95), np.max(cap_lens))
    # print('cost', '%.2f(min)' % ((time.time() - start_time) / 60))
    # print('num_data', len(dataset), 'num_batch', len(dataloader))
    # print('num_vps', len(dataloader.dataset._view_fts))
    # print('num_objs', np.sum([len(x) for x in dataloader.dataset.gt_obj_annos.values()]))

    # 4 20.30947831072043 33.0 62
    # cost 0.12(min)
    # num_data 10466 num_batch 82
    # num_vps 1537
    # num_objs 28038


