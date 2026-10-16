from.base_dataset import BaseMapDataset
from .map_utils.nuscmap_extractor import NuscMapExtractor
from mmdet.datasets import DATASETS
import numpy as np
from .visualize.renderer import Renderer
import mmcv
from time import time
from pyquaternion import Quaternion
import pickle
from copy import deepcopy
import pickle as pkl
from nmp.map_tiles.lane_render import \
    load_nusc_data_infos, \
    load_nusc_data_cities, \
    split_nusc_geo_bs, \
    get_token2map_index, \
    map_geo_loc, \
    creat_map_gpu_by_name, \
    map_tile_bound, \
    get_coords_resample, \
    resample_by_geo
from plugin.configs.instgpmap.nuscenes_oldsplit.instgpmap_nusc_oldsplit import SMALL_BATCH_DEBUG

@DATASETS.register_module()
class NuscDataset(BaseMapDataset):
    """NuScenes map dataset class.

    Args:
        ann_file (str): annotation file path
        cat2id (dict): category to class id
        roi_size (tuple): bev range
        eval_config (Config): evaluation config
        meta (dict): meta information
        pipeline (Config): data processing pipeline config
        interval (int): annotation load interval
        work_dir (str): path to work dir
        test_mode (bool): whether in test mode
    """
    
    def __init__(self, data_root, **kwargs):
        super().__init__(**kwargs)
        self.map_extractor = NuscMapExtractor(data_root, self.roi_size)
        self.renderer = Renderer(self.cat2id, self.roi_size, 'nusc')
    
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        
        start_time = time()
        ann = mmcv.load(ann_file)
        samples = ann[::self.interval]
        if SMALL_BATCH_DEBUG:
            pass
            # gpu0_sampler = mmcv.load("datasets/nuscenes/gpu0_sampler.pkl")
            # gpu0_sampler = mmcv.load("datasets/nuscenes/gpu0_sampler_train.pkl")
            # samples = [samples[i] for i in gpu0_sampler]
            # samples = ann[0:200]
        print(f'collected {len(samples)} samples in {(time() - start_time):.2f}s')
        self.samples = samples
        self.load_annotations_nmp(ann_file)
    
    def load_matching(self, matching_file):
        with open(matching_file, 'rb') as pf:
            data = pickle.load(pf)
        total_samples = 0
        for scene_name, info in data.items():
            total_samples += len(info['sample_ids'])
        # assert total_samples == len(self.samples), 'Matching info not matched with data samples'
        self.matching_meta = data
        print(f'loaded matching meta for {len(data)} scenes')


    def load_annotations_nmp(self, ann_file):
        if (hasattr(self, 'data_info_boston') and self.data_info_boston) or \
                (hasattr(self, 'nusc_city_split') and self.nusc_city_split) or \
                hasattr(self, 'nusc_new_split') and self.nusc_new_split:
            temp_ann_file = ['./data/nuscenes/nuScences_map_trainval_infos_train.pkl',
                             './data/nuscenes/nuScences_map_trainval_infos_val.pkl']
            data = [mmcv.load(ann) for ann in temp_ann_file]
            data_infos = list(data[0]['infos']) + list(data[1]['infos'])
        else:
            data_infos = mmcv.load(ann_file)
            # data_infos = list(data['infos'])

        if 'infos_train' in ann_file and not hasattr(self, 'sort_train_infos'):
            if hasattr(self, 'nusc_city_split') and self.nusc_city_split:
                new_train_infos = []
                new_train_data_cities = []
                train_infos, train_data_cities = self.gen_token2map_info('train')
                val_infos, val_data_cities = self.gen_token2map_info('val')
                data_infos = train_infos + val_infos
                data_cities = train_data_cities + val_data_cities
                for info, city_name in zip(data_infos, data_cities):
                    if city_name == 'singapore-onenorth' or \
                            city_name == 'boston-seaport':
                        new_train_infos.append(deepcopy(info))
                        new_train_data_cities.append(city_name)
                data_infos = new_train_infos
                data_city_names = new_train_data_cities

                print('dataset train data_infos', len(data_infos))
                print('dataset train data_city_names', len(data_city_names))

            elif hasattr(self, 'nusc_new_split') and self.nusc_new_split:
                new_train_infos = []
                new_train_data_cities = []
                train_infos, train_data_cities = self.gen_token2map_info('train')
                val_infos, val_data_cities = self.gen_token2map_info('val')
                data_infos = train_infos + val_infos
                data_cities = train_data_cities + val_data_cities
                with open(
                        f'/oldhome/xiongx/repository/neural_map_prior_code/project/train_sample_tokens.pkl', 'rb') as f:
                    sample_tokens_train = pkl.load(f)
                for info, city_name in zip(data_infos, data_cities):
                    if info['token'] in sample_tokens_train:
                        new_train_infos.append(deepcopy(info))
                        new_train_data_cities.append(city_name)
                data_infos = new_train_infos
                data_city_names = new_train_data_cities

                print('dataset train data_infos', len(data_infos))
                print('dataset train data_city_names', len(data_city_names))

        if 'infos_val' in ann_file and not hasattr(self, 'sort_val_infos'):
            if hasattr(self, 'nusc_city_split') and self.nusc_city_split:
                new_val_infos = []
                new_val_data_cities = []
                train_infos, train_data_cities = self.gen_token2map_info('train')
                val_infos, val_data_cities = self.gen_token2map_info('val')
                data_infos = train_infos + val_infos
                data_cities = train_data_cities + val_data_cities
                for info, city_name in zip(data_infos, data_cities):
                    if city_name == 'singapore-queenstown' or \
                            city_name == 'singapore-hollandvillage':
                        new_val_infos.append(deepcopy(info))
                        new_val_data_cities.append(city_name)
                data_infos = new_val_infos
                data_city_names = new_val_data_cities

                print('dataset val data_infos', len(data_infos))
                print('dataset val data_city_names', len(data_city_names))

            elif hasattr(self, 'nusc_new_split') and self.nusc_new_split:
                new_val_infos = []
                new_val_data_cities = []
                train_infos, train_data_cities = self.gen_token2map_info('train')
                val_infos, val_data_cities = self.gen_token2map_info('val')
                data_infos = train_infos + val_infos
                data_cities = train_data_cities + val_data_cities
                with open(
                        f'/oldhome/xiongx/repository/neural_map_prior_code/project/val_sample_tokens.pkl', 'rb') as f:
                    sample_tokens_val = pkl.load(f)
                for info, city_name in zip(data_infos, data_cities):
                    if info['token'] in sample_tokens_val:
                        new_val_infos.append(deepcopy(info))
                        new_val_data_cities.append(city_name)
                data_infos = new_val_infos
                data_city_names = new_val_data_cities

                print('dataset val data_infos', len(data_infos))
                print('dataset val data_city_names', len(data_city_names))

        if 'infos_train' in ann_file:
            dataset = 'train'
            if hasattr(self, 'sort_train_infos') and self.sort_train_infos:
                if hasattr(self, 'nusc_city_split') and self.nusc_city_split:
                    new_train_infos = []
                    new_train_data_cities = []
                    train_infos, train_data_cities = self.gen_token2map_info('train')
                    val_infos, val_data_cities = self.gen_token2map_info('val')
                    data_infos = train_infos + val_infos
                    data_cities = train_data_cities + val_data_cities
                    for info, city_name in zip(data_infos, data_cities):
                        if city_name == 'singapore-onenorth' or \
                                city_name == 'boston-seaport':
                            new_train_infos.append(deepcopy(info))
                            new_train_data_cities.append(city_name)
                    data_infos = new_train_infos
                    data_city_names = new_train_data_cities

                elif hasattr(self, 'nusc_new_split') and self.nusc_new_split:
                    new_train_infos = []
                    new_train_data_cities = []
                    train_infos, train_data_cities = self.gen_token2map_info('train')
                    val_infos, val_data_cities = self.gen_token2map_info('val')
                    data_infos = train_infos + val_infos
                    data_cities = train_data_cities + val_data_cities
                    with open(
                            f'/oldhome/xiongx/repository/neural_map_prior_code/project/train_sample_tokens.pkl',
                            'rb') as f:
                        sample_tokens_train = pkl.load(f)
                    for info, city_name in zip(data_infos, data_cities):
                        if info['token'] in sample_tokens_train:
                            new_train_infos.append(deepcopy(info))
                            new_train_data_cities.append(city_name)
                    data_infos = new_train_infos
                    data_city_names = new_train_data_cities

                else:
                    train_infos, train_data_cities = self.gen_token2map_info('train')
                    data_infos = train_infos
                    data_city_names = train_data_cities

                print('dataset train data_infos', len(data_infos))
                print('dataset train data_city_names', len(data_city_names))

                if (hasattr(self, 'nusc_city_split') and self.nusc_city_split) or (
                        hasattr(self, 'nusc_new_split') and self.nusc_new_split):
                    dataset = 'city'
                else:
                    dataset = 'train'
                nusc_train_min_geo_loc, nusc_train_max_geo_loc = map_geo_loc(dataset)
                train_token2map_index, train_map_index2token = \
                    get_token2map_index(
                        data_infos,
                        data_city_names,
                        nusc_train_min_geo_loc,
                        nusc_train_max_geo_loc,
                        map_prior_tile_size=self.gm_grid_size
                    )
                self.token2map_index,self.map_index2token = train_token2map_index, train_map_index2token 
                self.train_gpu2city = split_nusc_geo_bs(#(map_tile,gpu_id)
                    self.sort_batch_size,
                    data_infos,
                    data_city_names,
                    dataset,
                    train_token2map_index,
                    self.gm_grid_size,
                    train_map_index2token)
                self.gpu2sample_id = resample_by_geo(
                    self.sort_batch_size,
                    data_infos,
                    data_city_names,
                    dataset,
                    self.train_gpu2city,
                    train_token2map_index
                )

        elif 'infos_val' in ann_file:
            dataset = 'val'
            if hasattr(self, 'sort_val_infos') and self.sort_val_infos:
                if hasattr(self, 'nusc_city_split') and self.nusc_city_split:
                    new_val_infos = []
                    new_val_data_cities = []
                    train_infos, train_data_cities = self.gen_token2map_info('train')
                    val_infos, val_data_cities = self.gen_token2map_info('val')
                    data_infos = train_infos + val_infos
                    data_cities = train_data_cities + val_data_cities
                    for info, city_name in zip(data_infos, data_cities):
                        if city_name == 'singapore-queenstown' or \
                                city_name == 'singapore-hollandvillage':
                            new_val_infos.append(deepcopy(info))
                            new_val_data_cities.append(city_name)
                    data_infos = new_val_infos
                    data_city_names = new_val_data_cities

                elif hasattr(self, 'nusc_new_split') and self.nusc_new_split:
                    new_val_infos = []
                    new_val_data_cities = []
                    train_infos, train_data_cities = self.gen_token2map_info('train')
                    val_infos, val_data_cities = self.gen_token2map_info('val')
                    data_infos = train_infos + val_infos
                    data_cities = train_data_cities + val_data_cities
                    with open(
                            f'/oldhome/xiongx/repository/neural_map_prior_code/project/val_sample_tokens.pkl',
                            'rb') as f:
                        sample_tokens_val = pkl.load(f)
                    for info, city_name in zip(data_infos, data_cities):
                        if info['token'] in sample_tokens_val:
                            new_val_infos.append(deepcopy(info))
                            new_val_data_cities.append(city_name)
                    data_infos = new_val_infos
                    data_city_names = new_val_data_cities

                else:
                    val_infos, val_data_cities = self.gen_token2map_info('val')
                    data_infos = val_infos
                    data_city_names = val_data_cities

                print('dataset val data_infos', len(data_infos))
                print('dataset val data_city_names', len(data_city_names))

                if (hasattr(self, 'nusc_city_split') and self.nusc_city_split) or (
                        hasattr(self, 'nusc_new_split') and self.nusc_new_split):
                    dataset = 'city'
                else:
                    dataset = 'val'
                nusc_val_min_geo_loc, nusc_val_max_geo_loc = map_geo_loc(dataset)
                val_token2map_index, val_map_index2token = \
                    get_token2map_index(
                        data_infos,
                        data_city_names,
                        nusc_val_min_geo_loc,
                        nusc_val_max_geo_loc,
                        map_prior_tile_size=self.gm_grid_size
                    )
                self.token2map_index,self.map_index2token = val_token2map_index, val_map_index2token 
                self.val_gpu2city = split_nusc_geo_bs(
                    self.sort_batch_size,
                    data_infos,
                    data_city_names,
                    dataset,
                    val_token2map_index,
                    self.gm_grid_size,
                    val_map_index2token
                )
                self.gpu2sample_id = resample_by_geo(
                    self.sort_batch_size,
                    data_infos,
                    data_city_names,
                    dataset,
                    self.val_gpu2city,
                    val_token2map_index
                )
        else:
            dataset = None
            

        # data_infos = data_infos[::self.interval]
        print('aaaa')

    def gen_token2map_info(self, split):
        data_infos = load_nusc_data_infos(split)
        data_cities = load_nusc_data_cities(split)

        if hasattr(self, 'sort_batch_size') and not (self.sort_batch_size == 1):
            tail = len(data_infos) % self.sort_batch_size
            if tail != 0:
                data_infos = data_infos[:-tail]
                data_cities = data_cities[:-tail]
            assert len(data_infos) == len(data_cities), \
                (len(data_infos), len(data_cities))

        return data_infos, data_cities
    
    def get_sample(self, idx):
        """Get data sample. For each sample, map extractor will be applied to extract 
        map elements. 

        Args:
            idx (int): data index

        Returns:
            result (dict): dict of input
        """

        sample = self.samples[idx]
        location = sample['location']

        lidar2ego = np.eye(4)
        lidar2ego[:3,:3] = Quaternion(sample['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = sample['lidar2ego_translation']

        ego2global = np.eye(4)
        ego2global[:3,:3] = Quaternion(sample['e2g_rotation']).rotation_matrix
        ego2global[:3, 3] = sample['e2g_translation']

        # NOTE: The original StreamMapNet uses the ego location to query the map,
        # to align with the lidar-centered setting in MapTR, we made some modifiactions 
        # here to switch to the lidar-center setting
        lidar2global = ego2global @ lidar2ego
        lidar2global_translation = list(lidar2global[:3, 3])
        lidar2global_translation = [float(x) for x in lidar2global_translation]
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

        map_geoms = self.map_extractor.get_map_geom(location, lidar2global_translation, 
                lidar2global_rotation)
        
        lidar_shifted_e2g_translation = np.array(sample['e2g_translation'])
        lidar_shifted_e2g_translation[0] = lidar2global_translation[0]
        lidar_shifted_e2g_translation[1] = lidar2global_translation[1]
        lidar_shifted_e2g_translation = lidar_shifted_e2g_translation.tolist()
        e2g_rotation = sample['e2g_rotation']

        lidar2global = np.eye(4)
        lidar2global[:3,:3] = Quaternion(e2g_rotation).rotation_matrix
        lidar2global[:3, 3] = lidar_shifted_e2g_translation
        global2lidar = np.linalg.inv(lidar2global)
        
        ego2lidar = global2lidar  @ ego2global

        map_label2geom = {}
        for k, v in map_geoms.items():
            if k in self.cat2id.keys():
                map_label2geom[self.cat2id[k]] = v
        
        ego2img_rts = []
        ego2cam_rts = []
        for c in sample['cams'].values():
            extrinsic, intrinsic = np.array(
                c['extrinsics']), np.array(c['intrinsics'])

            # ego coord to cam coord
            #ego2cam_rt = extrinsic

            cam2ego_rt = np.linalg.inv(extrinsic)
            cam2lidar_rt = ego2lidar @ cam2ego_rt
            lidar2cam_rt = np.linalg.inv(cam2lidar_rt)
            ego2cam_rt = lidar2cam_rt

            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

            ego2img_rt = (viewpad @ ego2cam_rt)
            ego2cam_rts.append(ego2cam_rt)
            ego2img_rts.append(ego2img_rt)


        input_dict = {
            'location': location,
            'token': sample['token'],
            'img_filenames': [c['img_fpath'] for c in sample['cams'].values()],
            # intrinsics are 3x3 Ks
            'cam_intrinsics': [c['intrinsics'] for c in sample['cams'].values()],
            # extrinsics are 4x4 tranform matrix, **ego2cam**
            'cam_extrinsics': [c['extrinsics'] for c in sample['cams'].values()],
            'ego2img': ego2img_rts,
            'ego2cam': ego2cam_rts,
            'map_geoms': map_label2geom, # {0: List[ped_crossing(LineString)], 1: ...}
            #'ego2global_translation': sample['e2g_translation'], 
            #'ego2global_rotation': Quaternion(sample['e2g_rotation']).rotation_matrix.tolist(),
            'ego2global_translation': lidar_shifted_e2g_translation, 
            'ego2global_rotation': Quaternion(e2g_rotation).rotation_matrix.tolist(),
            'sample_idx': sample['sample_idx'],
            'scene_name': sample['scene_name'],
            'lidar2ego_translation': sample['lidar2ego_translation'],
            'lidar2ego_rotation': sample['lidar2ego_rotation'],
        }

        return input_dict