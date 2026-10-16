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
    get_coords_resample
from collections import defaultdict, OrderedDict
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import torch
import numpy as np
import matplotlib.pyplot as plt
from tools.visualization.vis_global import merge_boundary, merge_divider, merge_corssing, instance_bank2mask,draw_instance_bank2mask,global_points_bank2mask
from tools.tracking.prepare_gt_tracks import draw_instance_masks_pred,interp_fixed_num,find_matchings
from shapely.geometry import LineString, Point
from itertools import chain
import os
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box, MultiLineString
from typing import Dict, List, Tuple, Optional, Union
from plugin.datasets.pipelines.vectorize import VectorizeMap
class line_cfg():
    def __init__(self) -> None:
        self.simplify = 0.5
        self.line_opacity = 0.5
from plugin.configs.instgpmap.nuscenes_oldsplit.instgpmap_nusc_oldsplit import DEBUG
from mmcv.runner import get_dist_info
import mmcv
import time
from mmdet.core import multi_apply
from nmp.vis_debug import vis_line
from global_parameter import thr_track,thr_det_global,thr_matching_mask_label0,thr_matching_mask_label12,thr_pos_rm,single_traversal,map_add,map_rm,map_change
# 自定义排序函数
def custom_sort_key(key):
    a, b = key.split('_')
    return (int(a), int(b))

class MapGlobalMemory(object):
    def __init__(self, map_attribute, bev_attribute,use_memory_bank_new,PRIOR_MODE,mem_select_dist_ranges):
        self.bev_attribute = bev_attribute
        self.map_attribute = map_attribute
        self.use_memory_bank_new = use_memory_bank_new
        self.PRIOR_MODE = PRIOR_MODE

        if 'nusc_city_split' in map_attribute and map_attribute['nusc_city_split']:
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
            train_infos = new_train_infos
            train_data_cities = new_train_data_cities

        elif 'nusc_new_split' in map_attribute and map_attribute['nusc_new_split']:
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
            train_infos = new_train_infos
            train_data_cities = new_train_data_cities

        else:
            train_infos, train_data_cities = self.gen_token2map_info('train')

        print('training new_data_infos', len(train_infos))
        print('training new_data_city', len(train_data_cities))
        if ('nusc_city_split' in map_attribute and map_attribute['nusc_city_split']) or (
                'nusc_new_split' in map_attribute and map_attribute['nusc_new_split']):
            dataset = 'city'
        else:
            dataset = 'train'
        self.nusc_train_min_geo_loc, self.nusc_train_max_geo_loc = \
            map_geo_loc(dataset)
        self.train_token2map_index, self.train_map_index2token = \
            get_token2map_index(
                train_infos,
                train_data_cities,
                self.nusc_train_min_geo_loc,
                self.nusc_train_max_geo_loc,
                map_attribute['global_map_tile_size'])
        self.train_gpu2city = split_nusc_geo_bs(#是一个列表,存储的是每张gpu分到的tile是哪几个
            self.map_attribute['batch_size'],
            train_infos,
            train_data_cities,
            dataset,
            self.train_token2map_index,
            self.map_attribute['global_map_tile_size'],
            self.train_map_index2token)

        if 'nusc_city_split' in map_attribute and map_attribute['nusc_city_split']:
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
            val_infos = new_val_infos
            val_data_cities = new_val_data_cities

        elif 'nusc_new_split' in map_attribute and map_attribute['nusc_new_split']:
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
            val_infos = new_val_infos
            val_data_cities = new_val_data_cities

        else:
            val_infos, val_data_cities = self.gen_token2map_info('val')

        print('training new_data_infos', len(val_infos))
        print('training new_data_city', len(val_data_cities))
        if ('nusc_city_split' in map_attribute and map_attribute['nusc_city_split']) or (
                'nusc_new_split' in map_attribute and map_attribute['nusc_new_split']):
            dataset = 'city'
        else:
            dataset = 'val'
        self.nusc_val_min_geo_loc, self.nusc_val_max_geo_loc = \
            map_geo_loc(dataset)
        self.val_token2map_index, self.val_map_index2token = \
            get_token2map_index(
                val_infos,
                val_data_cities,
                self.nusc_val_min_geo_loc,
                self.nusc_val_max_geo_loc,
                self.map_attribute['global_map_tile_size'])
        self.val_gpu2city = split_nusc_geo_bs(
            self.map_attribute['batch_size'],
            val_infos,
            val_data_cities,
            dataset,
            self.val_token2map_index,
            self.map_attribute['global_map_tile_size'],
            self.val_map_index2token)

        self.map_slice_float_dict = defaultdict(dict) #每张gpu的程序,该字典会有不同的tile
        self.map_slice_int_dict = {}
        self.map_center_dis_dict = {}
        self.map_slice_onehot_dict = {}

        self.train_epoch_point = -2
        self.val_epoch_point = -2
        self.agrs = line_cfg()
        self.patch_size = [60,30]
        self.VectorizeMap = VectorizeMap(        
            coords_dim=2,
            roi_size=(60, 30),
            sample_num=20,
            normalize=True,
            permute=False,)
        self.plot_interval = 2000*3 if not DEBUG else 150
        self.plot_index = 0
        self.use_juzheng_trans = True
        # self.mem_select_dist_ranges = [0, 1, 5, 10, 15]
        self.mem_select_dist_ranges = mem_select_dist_ranges
    def gen_token2map_info(self, split):
        data_infos = load_nusc_data_infos(split)
        data_cities = load_nusc_data_cities(split)

        # if not (self.map_attribute['batch_size'] == 1):
        #     tail = len(data_infos) % self.map_attribute['batch_size']
        #     if tail != 0:
        #         data_infos = data_infos[:-tail]
        #         data_cities = data_cities[:-tail]
        #     assert len(data_infos) == len(data_cities), \
        #         (len(data_infos), len(data_cities))

        return data_infos, data_cities

    @staticmethod
    def gen_map_slice_int(map_attribute):
        map_attribute = deepcopy(map_attribute)
        tile_param = map_attribute['tile_param']
        tile_param['data_type'] = 'int16'
        tile_param['embed_dims'] = 1
        return map_attribute

    def reset_define_map(self, epoch, gpu_id, dataset, map_slices_name, map_attribute_func=None,sample_idx=None):
        map_attribute = deepcopy(self.map_attribute)
        if self.check_epoch(epoch, dataset):#只有在每个epoch的第一个iter初始化全局地图

            if isinstance(map_slices_name, str):
                map_slice_dict = getattr(self, map_slices_name)
                if map_attribute_func is not None:
                    map_attribute = map_attribute_func[map_slices_name](map_attribute)
                self.reset_map(epoch, gpu_id, dataset, map_slice_dict, map_attribute)
                print(f'create empty map for: {map_slices_name}, {map_attribute}')

            elif isinstance(map_slices_name, list):
                for map_name in map_slices_name:
                    map_slice_dict = getattr(self, map_name)
                    self.reset_map(epoch, gpu_id, dataset, map_slice_dict, map_attribute)
                    print(f'create empty map for: {map_name}, {map_attribute}')

    def check_epoch(self, epoch, dataset):
        if ((self.train_epoch_point != epoch) and (dataset == 'train')) or \
                ((self.val_epoch_point != epoch) and (dataset == 'val')):

            if dataset == 'train':
                print('epoch:', epoch, 'self.train_epoch_point:', self.train_epoch_point)
                self.train_epoch_point = epoch
                return True
            elif dataset == 'val':
                print('epoch:', epoch, 'self.val_epoch_point:', self.val_epoch_point)
                self.val_epoch_point = epoch
                return True

    def reset_map(self, epoch, gpu_id, dataset, map_slices_dict, map_attribute):
        gpu_city_list_all_gpu  = []
        for i in range(len(getattr(self, f'{dataset}_gpu2city'))):
            gpu_city_list = [city_name for (city_name, city_pop_sample) in getattr(self, f'{dataset}_gpu2city')[i]]#city_pop_sample:这个tile拥有的帧数
            gpu_city_list_all_gpu.extend(gpu_city_list)
        min_geo_loc = getattr(self, f'nusc_{dataset}_min_geo_loc')
        max_geo_loc = getattr(self, f'nusc_{dataset}_max_geo_loc')
        # if dataset=='train' and epoch==0:
        #     os.makedirs('../map_slices_dict',exist_ok=True)
        #     mmcv.dump(map_slices_dict,f'../map_slices_dict/epoch{epoch}_{gpu_id}.pkl')
        # for k in map_slices_dict.keys():
        #     map_slices_dict[k]['instance_bank'].clear()
        #     map_slices_dict[k]['global_instance_bank'].clear()
        map_slices_dict.clear()

        creat_map_gpu_by_name(
            map_slices_dict,
            map_attribute,
            min_geo_loc, max_geo_loc,
            gpu_city_list=gpu_city_list_all_gpu)
        # if dataset == 'train':
        #     import copy
        #     map_slices_dict_saved = copy.deepcopy(mmcv.load(f'../map_slices_dict/epoch0_{gpu_id}.pkl'))
        #     for tile_name,values in map_slices_dict_saved.items():
        #         map_slices_dict[tile_name] = values

        # print('reset map map_slices_dict:', map_slices_dict)

    def gen_map_info(self, token, split):
        city_name = getattr(self, f'{split}_token2map_index')[token]['loc']
        map_index = getattr(self, f'{split}_token2map_index')[token]['map_index_list'][0]
        map_slice_min_bound, map_slice_max_bound = map_tile_bound(
            getattr(self, f'nusc_{split}_min_geo_loc'),
            getattr(self, f'nusc_{split}_max_geo_loc'),
            city_name,
            map_index,
            self.map_attribute['global_map_tile_size'])
        return city_name, map_index, map_slice_min_bound, map_slice_max_bound

    def get_inverse_matrix(self,prev2curr_matrix):
        # 假设 prev2curr_matrix 是给定的 4x4 齐次变换矩阵
        R = prev2curr_matrix[:3, :3]  # 提取旋转矩阵 (3x3)
        t = prev2curr_matrix[:3, 3]   # 提取平移向量 (3x1)
        # 计算旋转矩阵的转置
        R_inv = R.T  # 或者 R.transpose(0, 1)
        # 计算新的平移向量
        t_inv = -R_inv @ t
        # 构造逆变换矩阵
        inv_matrix = torch.eye(4, dtype=prev2curr_matrix.dtype)  # 初始化为单位矩阵
        inv_matrix[:3, :3] = R_inv  # 设置旋转部分
        inv_matrix[:3, 3] = t_inv   # 设置平移部分
        return inv_matrix
    
    def take_map_prior(self, img_meta, dataset,update_global_instance_flag):
        token = img_meta['token']
        city_name, map_index, map_slice_min_bound, map_slice_max_bound = self.gen_map_info(token, dataset)
        # map_index = np.array([1,2])
        global_map_slice = self.map_slice_float_dict[f'map_{city_name}_{map_index[0]}_{map_index[1]}']

        prior = defaultdict(list)

        map_pose = img_meta['ego2global_translation'][:2]
        rotation = Quaternion(matrix=np.array(img_meta['ego2global_rotation']))
        patch_box = (map_pose[0], map_pose[1],  self.patch_size[1],self.patch_size[0])#TODO:之后弄清楚这里为啥要反过来,相对于gen_vectorized_samples函数,原因大概率是那里用的rotation是lidar2global,这里是ego2global
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        patch = get_patch_coord(patch_box, patch_angle)
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        instance_bank,global_instance_bank = global_map_slice['instance_bank'],global_map_slice['global_instance_bank']
        bank_list = global_map_slice['bank_list']
        if update_global_instance_flag and single_traversal:
            bank_list.clear()

        prev_e2g_trans =  torch.tensor(img_meta['ego2global_translation'], dtype=torch.float64)
        prev_e2g_rot = torch.tensor(img_meta['ego2global_rotation'], dtype=torch.float64)
        prev_e2g_matrix = torch.eye(4, dtype=torch.float64)
        prev_e2g_matrix[:3, :3] = prev_e2g_rot
        prev_e2g_matrix[:3, 3] = prev_e2g_trans
        prev2curr_matrix = prev_e2g_matrix
        inv_matrix = self.get_inverse_matrix(prev2curr_matrix)
    
        #使用bank_list判断是该global_id在该帧内部
        for vec_tag, polylines in bank_list.items():
            bank_list_values =[np.array(value[0]) for value in list(bank_list[vec_tag].values()) if not isinstance(value, int)]
            bank_list_values = np.array(bank_list_values)
            prev2curr_bank_list_values = get_prev2curr_vectors(bank_list_values, inv_matrix,[-30., -15.],(60, 30),False,result_normalize=True,clip=False)
            mask_x = (prev2curr_bank_list_values[:,:,0] > 0) & (prev2curr_bank_list_values[:,:,0] < 1)
            mask_y = (prev2curr_bank_list_values[:,:,1] > 0) & (prev2curr_bank_list_values[:,:,1] < 1)
            masks = mask_x & mask_y#必定有1个点在范围里面才为True
            intersection_flag = masks.any(axis=1)
            if intersection_flag.any():
                prior[vec_tag] = np.array(intersection_flag)


        #使用最近帧的先验
        global_points = []
        global_points_scores = []
        bank_points_list = []
        bank_points_scores_list = []
        bank_points_prev2curr_pred_vectors_list = []
        for vec_tag in prior.keys():
            # all_translation = np.array([value[1] for value in bank_list[vec_tag].values])
            all_translation = np.array([key for key in bank_list[vec_tag].keys() if key != 'rm_times'])
            dists = np.linalg.norm(np.array(img_meta['ego2global_translation'])[None, :][:,:2] - all_translation[:,:2], axis=1)
            dists_order_index = np.argsort(dists)#距离最小的排在最前面

            intersection_flag = prior[vec_tag]
            all_translation_intersection_pos = all_translation[intersection_flag]
            #训练时候的抽选某个global_id的query应该是随机抽,推理是按照距离抽,但是track_query不管训练和推理都是是离的最近的那个
            select_indices = self.select_memory_entries(all_translation_intersection_pos,np.array(img_meta['ego2global_translation'])[:2])#最近的在最后一个
            # select_indices[:-1]为memory_corss_attn抛弃最后一个
            selected_indices_keys = [all_translation_intersection_pos[i] for i in select_indices]
            bank_points =  [bank_list[vec_tag][tuple(key)][0] for key in selected_indices_keys]
            bank_points_scores = [bank_list[vec_tag][tuple(key)][1] for key in selected_indices_keys]
            bank_points_list.append(bank_points)
            bank_points_scores_list.append(bank_points_scores)

            # dists_order_index[0] = -1#改变,使用上一帧的先验            
            global_points.append(bank_list[vec_tag][tuple(all_translation[dists_order_index[0]])][0])
            global_points_scores.append(bank_list[vec_tag][tuple(all_translation[dists_order_index[0]])][1])
            #使用select_memory_entries中选择距离,选中距离0最近的,作为最近的先验,应该和上面这句作用一样
            # global_points.append(bank_list[vec_tag][tuple(all_translation_intersection_pos[select_indices[-1]])][0])

        for bank_points in bank_points_list:
            if self.use_memory_bank_new:
                bank_points_prev2curr_pred_vectors = get_prev2curr_vectors(bank_points, inv_matrix,[-30., -15.],(60, 30),False,result_normalize=True,clip=False)
                bank_points_prev2curr_pred_vectors_list.append(bank_points_prev2curr_pred_vectors)
            else:
                bank_points_prev2curr_pred_vectors_list.append([])
                

        # global_points = np.array(global_points)
        prev2curr_pred_vectors = get_prev2curr_vectors(global_points, inv_matrix,[-30., -15.],(60, 30),False,result_normalize=True,clip=True)
        prior_vector = defaultdict(list)#每个global_id,[[],[]]分别代表用于track_query和memory_bank_corss_attn的元素
        prior_vector_scores = defaultdict(list)
        {prior_vector.update({key: [np.array(prev2curr_pred_vectors[index]),bank_points_prev2curr_pred_vectors_list[index]]}) for index,(key, value) in enumerate(prior.items())}
        {prior_vector_scores.update({key: [global_points_scores[index],bank_points_scores_list[index]]}) for index,(key, value) in enumerate(prior.items())}
        #计算提取到当前帧的这些global_id的相似度合并逻辑
        if len(global_points) and dataset == 'val':
            x_min,x_max,y_min,y_max = np.stack(global_points)[:,:,0].min(),np.stack(global_points)[:,:,0].max(),np.stack(global_points)[:,:,1].min(),np.stack(global_points)[:,:,1].max()
            instance_bank_masks,instance_bank_masks_id = global_points_bank2mask(global_points,x_min,y_min,x_max,y_max,list(prior.keys()))
            matchings_instance_bank= find_matchings(instance_bank_masks,instance_bank_masks, thresh=thr_matching_mask_label12,scene_matcing=True,thershold_label0=thr_matching_mask_label0)
            #对于instance_bank中不同global_id的元素,使用后处理进行合并
            for label, matching  in matchings_instance_bank.items():
                pre_matching,cur_matching = matching
                for i in range(len(pre_matching)):
                    if pre_matching[i] != -1:
                        pre_matching[pre_matching[i]] = -1#防止遍历两次,因为是同一个instance_bank的元素
                        pre_tag = instance_bank_masks_id[label][i]
                        cur_tag = instance_bank_masks_id[label][pre_matching[i]]
                        #处理bank_list
                        pre2cur_vectors = {}
                        for x_y in set(list(bank_list[pre_tag].keys())+list(bank_list[cur_tag].keys())):
                            if x_y == 'rm_times':
                                continue
                            if x_y in bank_list[pre_tag] and x_y in bank_list[cur_tag]:
                                if bank_list[pre_tag][x_y][1] >= bank_list[cur_tag][x_y][1]:
                                    pre2cur_vectors[x_y] = bank_list[pre_tag][x_y]
                                else:
                                    pre2cur_vectors[x_y] = bank_list[cur_tag][x_y]
                            else:
                                if x_y in bank_list[pre_tag]:
                                    pre2cur_vectors[x_y] = bank_list[pre_tag][x_y]
                                else:
                                    pre2cur_vectors[x_y] = bank_list[cur_tag][x_y]
                        bank_list[cur_tag] = pre2cur_vectors
                        bank_list.pop(pre_tag)
                        prior_vector.pop(pre_tag)
                        prior_vector_scores.pop(pre_tag)

            # draw_instance_bank2mask(instance_bank_masks,f'masks/{self.plot_index}')
        sorted_keys = sorted(prior_vector.keys(),key=custom_sort_key)
        # 根据排序后的键顺序创建一个新的有序字典
        sorted_prior = OrderedDict((key, prior_vector[key]) for key in sorted_keys)
        sorted_prior_scores = OrderedDict((key, prior_vector_scores[key]) for key in sorted_keys)
        return [sorted_prior,sorted_prior_scores,bank_list]

    def select_memory_entries(self, mem_trans, curr_meta):
        history_e2g_trans = mem_trans
        curr_e2g_trans = curr_meta
        dists = np.linalg.norm(history_e2g_trans - curr_e2g_trans[None, :], axis=1)

        sorted_indices = np.argsort(dists)
        sorted_dists = dists[sorted_indices]
        covered = np.zeros_like(sorted_indices).astype(np.bool)
        selected_ids = []
        for dist_range in self.mem_select_dist_ranges[::-1]:
            outter_valid_flags = (sorted_dists >= dist_range) & ~covered
            if outter_valid_flags.any():
                pick_id = np.where(outter_valid_flags)[0][0]     
                covered[pick_id:] = True
            else:
                inner_valid_flags = (sorted_dists < dist_range) & ~covered
                if inner_valid_flags.any():
                    pick_id = np.where(inner_valid_flags)[0][-1]
                    covered[pick_id] = True
                else:
                    # return the mem_len closest one, but in the order of far -> close
                    return np.array(sorted_indices[:4][::-1])
            selected_ids.append(pick_id)

        selected_mem_ids = sorted_indices[np.array(selected_ids)]
        return selected_mem_ids
    
    def replace_map_prior(self, img_meta, dataset,pos_results,roi_size,origin,update_global_instance_flag,logger,pos_th=0.6,rm_prior_list=None):
        logger.info('11_1')
        token = img_meta['token']
        city_name, map_index, map_slice_min_bound, map_slice_max_bound = self.gen_map_info(token, dataset)
        # map_index = np.array([1,2])
        global_map_slice = self.map_slice_float_dict[f'map_{city_name}_{map_index[0]}_{map_index[1]}']

        logger.info('11_2')
        prev_e2g_trans =  torch.tensor(img_meta['ego2global_translation'], dtype=torch.float64)
        prev_e2g_rot = torch.tensor(img_meta['ego2global_rotation'], dtype=torch.float64)
        prev_e2g_matrix = torch.eye(4, dtype=torch.float64)
        prev_e2g_matrix[:3, :3] = prev_e2g_rot
        prev_e2g_matrix[:3, 3] = prev_e2g_trans
        prev2curr_matrix = prev_e2g_matrix
        if (len(np.array(pos_results["vectors"])) == 0):
            print('no vectors')
            return
        logger.info('11_3')
        vectors = np.array(pos_results["vectors"]).reshape((len(np.array(pos_results["vectors"])), 20, 2))
        prev2curr_pred_vectors = get_prev2curr_vectors(vectors, prev2curr_matrix,origin,roi_size,True)

        logger.info('11_4')
        vet_tags = [str(label)+"_"+str(vec_glb_idx)  for label,vec_glb_idx in zip(pos_results["labels"],pos_results["global_ids"])]
        instance_bank,global_instance_bank,bank_list = global_map_slice['instance_bank'],global_map_slice['global_instance_bank'],global_map_slice['bank_list']

        x_min = prev2curr_pred_vectors[...,0].min()
        x_max = prev2curr_pred_vectors[...,0].max()
        y_min = prev2curr_pred_vectors[...,1].min()
        y_max = prev2curr_pred_vectors[...,1].max()
        x_mid = img_meta['ego2global_translation'][0]
        y_mid =img_meta['ego2global_translation'][1]
        for vec_tag, polylines_dict in bank_list.items():
            polylines = [value[0] for value in polylines_dict.values() if not isinstance(value, int)]
            for polyline in polylines:
                one_line = np.array(polyline)
                pts = one_line[:, :2]
                x_min = min(x_min, pts[:, 0].min())
                x_max = max(x_max, pts[:, 0].max())
                y_min = min(y_min, pts[:, 1].min())
                y_max = max(y_max, pts[:, 1].max())


        logger.info('11_5')
        if self.PRIOR_MODE == 'v1' or (dataset == 'train' and self.PRIOR_MODE == 'v5'):
            pos_th = 1.2#不加先验的模式
        pos_th = 0.1
        pos_th_flag = pos_results['scores'] > pos_th
        len_pos = pos_th_flag.sum()
        rm_tag_list = []
        if rm_prior_list is not None and map_rm:
            for tag,times in rm_prior_list.items():
                if times >= 1:
                    rm_tag_list.append(tag)
            for rm_tag in rm_tag_list:
                rm_prior_list.pop(rm_tag)
                bank_list.pop(rm_tag)
                    
        keep_jingdu = 2
        for prev2curr_pred_vector,vec_tag,score in zip(prev2curr_pred_vectors[pos_th_flag],np.array(vet_tags)[pos_th_flag],pos_results['scores'][pos_th_flag]):
            if (round(img_meta['ego2global_translation'][0], keep_jingdu),round(img_meta['ego2global_translation'][1], keep_jingdu)) in bank_list:
                if score >= bank_list[vec_tag][(round(img_meta['ego2global_translation'][0], keep_jingdu),round(img_meta['ego2global_translation'][1], keep_jingdu))][1]:
                    bank_list[vec_tag][(round(img_meta['ego2global_translation'][0], keep_jingdu),round(img_meta['ego2global_translation'][1], keep_jingdu))] = [prev2curr_pred_vector,score]
            else:
                bank_list[vec_tag][(round(img_meta['ego2global_translation'][0], keep_jingdu),round(img_meta['ego2global_translation'][1], keep_jingdu))] = [prev2curr_pred_vector,score]

        #1.后处理matching2.更新instance_bank到global_instance_bank
        if dataset == 'val' and False:
            update_global_instance_bank(global_instance_bank,instance_bank,x_min,y_min,x_max,y_max,update_global_instance_flag,bank_list,plot_index=self.plot_index)
        
        # 可视化全局先验
        pred_save_folder = f'../global_intance_bank'
        frame_timestep = self.plot_index

        self.plot_index += 1
        if self.plot_index % self.plot_interval == 0 :
            plot_bank_list_distence_intervel10(bank_list,pred_save_folder,self.plot_index,x_min,y_min,x_max,y_max,f'map_{city_name}_{map_index[0]}_{map_index[1]}',x_mid,y_mid,dataset)

        #     plot_bank_list(bank_list,pred_save_folder,self.plot_index,x_min,y_min,x_max,y_max,f'map_{city_name}_{map_index[0]}_{map_index[1]}',x_mid,y_mid,dataset)
        #     # plot_instance_bank(instance_bank,global_instance_bank,pred_save_folder,self.plot_index,x_min,y_min,x_max,y_max,f'map_{city_name}_{map_index[0]}_{map_index[1]}',x_mid,y_mid)
        #     # vis_line(pred_save_folder+f'/{frame_timestep}.png',*instance_bank['1_5'])
        #     tile_name = f'map_{city_name}_{map_index[0]}_{map_index[1]}'
        #     save_global_dict = {
        #         'instance_bank': instance_bank,
        #         'global_instance_bank': global_instance_bank,
        #         'pred_save_folder': pred_save_folder,
        #         'frame_timestep': frame_timestep,
        #         'x_min': x_min,
        #         'y_min': y_min,
        #         'x_max': x_max,
        #         'y_max': y_max,
        #         'tile_name':tile_name
        #     }
        #     os.makedirs(f'../global_intance_bank',exist_ok=True)
        #     mmcv.dump(save_global_dict,f'../global_intance_bank/{frame_timestep}_{tile_name}.pkl')
        #     print(f'save to ../global_intance_bank/{frame_timestep}_{tile_name}.pkl')
        return [True]


    def process_instance(self,vec_tag, curr_vec,instance_bank,global_instance_bank):
        label, vec_glb_idx = vec_tag.split('_')
        label = int(label)

        vec_glb_idx = int(vec_glb_idx)
        curr_vec_polyline = LineString(curr_vec)

        if vec_tag not in instance_bank and vec_tag not in global_instance_bank: # if the instance first appears, instance_bank就是个存储的全局地图
            polylines = [curr_vec_polyline,]
            flag = 0
        else: # if the instance has appeared before, polylines = previous merged polyline + current polyline
            if vec_tag in instance_bank:
                polylines = instance_bank[vec_tag] + [curr_vec_polyline,]
                flag = 0
            else:
                polylines = global_instance_bank[vec_tag] + [curr_vec_polyline,]
                flag = 1
        
        if label == 0: # crossing, merged by convex hull
            polygon = merge_corssing(polylines)
            polygon = polygon.simplify(self.agrs.simplify)
            vector = np.array(polygon.exterior.coords) 

            updated_polyline = LineString(vector)
            updated_polylines = [updated_polyline, ]

        elif label == 1: # divider, merged fitting a polyline
            polylines_vecs = [np.array(one_line.coords) for one_line in polylines]
            polylines_vecs = merge_boundary(polylines_vecs)

            # update instance bank for line
            updated_polylines = [LineString(vec) for vec in polylines_vecs]

        elif label == 2: # boundary, do not merge
            polylines_vecs = [np.array(one_line.coords) for one_line in polylines]
            polylines_vecs = merge_boundary(polylines_vecs)

            # update instance bank for line
            updated_polylines = [LineString(vec) for vec in polylines_vecs]
            if flag == 0:
                instance_bank[vec_tag] = updated_polylines
            else:
                global_instance_bank[vec_tag] = updated_polylines
        
        return [updated_polylines,flag]



def plot_instance_bank(instance_bank,global_instance_bank,pred_save_folder,frame_timestep,x_min,y_min,x_max,y_max,tile_name,x_mid,y_mid):
    rank, world_size = get_dist_info()
    if rank != 0:
        return
    os.makedirs(pred_save_folder, exist_ok=True)
    plt.figure(facecolor='lightgreen')
    fig = plt.figure(figsize=(int(x_max - x_min + 10) ,int(y_max-y_min + 10)))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(float(x_min), float(x_max))
    ax.set_ylim(float(y_min), float(y_max))

    for vec_tag, polylines in chain(instance_bank.items(), global_instance_bank.items()):
        label, vec_glb_idx = vec_tag.split('_')
        label = int(label)
        vec_glb_idx = int(vec_glb_idx)

        if label == 0: # ped_crossing
            color = 'b'
        elif label == 1: # divider
            color = 'r'
        elif label == 2: # boundary
            color = 'g'

        if label == 0: # crossing, merged by convex hull
            vector = np.array(polylines[0].coords) 

            pts = vector[:, :2]
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=0.5)
            ax.plot(x, y, "o", color=color, markersize=50)
            # 在中间位置绘制tag
            mid_idx = len(pts) // 2
            ax.text(x.mean(),y.mean(), vec_tag, fontsize=12*5, ha='center', va='center', color='b')

        elif label == 1: # divider, merged fitting a polyline
            polylines_vecs = [np.array(line.coords) for line in polylines]

            for split_idx,one_line in enumerate(polylines_vecs):
                # one_line = np.array(LineString(one_line).simplify(args.simplify*2).coords)
                one_line = np.array(LineString(one_line).coords)
                pts = one_line[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=0.5)
                ax.plot(x, y, "o", color=color, markersize=50)
                # 在中间位置绘制tag
                mid_idx = len(pts) // 2
                ax.text(pts[mid_idx][0], pts[mid_idx][1], vec_tag+'_'+str(split_idx), fontsize=12*5, ha='center', va='center', color='black')

        elif label == 2: # boundary, do not merge
            polylines_vecs = [np.array(line.coords) for line in polylines]

            for split_idx,one_line in enumerate(polylines_vecs):
                # one_line = np.array(LineString(one_line).simplify(args.simplify).coords)
                one_line = np.array(LineString(one_line).coords)
                pts = one_line[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=0.5)
                ax.plot(x, y, "o", color=color, markersize=50)
                # 在中间位置绘制tag
                mid_idx = len(pts) // 2
                ax.text(pts[mid_idx][0], pts[mid_idx][1], vec_tag+'_'+str(split_idx), fontsize=12*5, ha='center', va='center', color='black')
    # 在中间位置绘制一个圆
    ax.plot(x_mid, y_mid, "o", color='black', markersize=50)
    pred_save_path = pred_save_folder + f'/{frame_timestep}_{tile_name}.png'
    plt.grid(False)
    plt.savefig(pred_save_path, bbox_inches='tight', transparent=False, dpi=20)
    plt.clf() 
    plt.close(fig)
    print("global_intance_bank  : ", pred_save_path)

def plot_bank_list(bank_list,pred_save_folder,frame_timestep,x_min,y_min,x_max,y_max,tile_name,x_mid,y_mid,dataset):
    rank, world_size = get_dist_info()
    if rank != 0:
        return
    os.makedirs(pred_save_folder+f'/{dataset}', exist_ok=True)
    plt.figure(facecolor='lightgreen')
    fig = plt.figure(figsize=(int(x_max - x_min + 10) ,int(y_max-y_min + 10)))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(float(x_min), float(x_max))
    ax.set_ylim(float(y_min), float(y_max))

    for vec_tag, polylines_dict in bank_list.items():
        polylines = [value[0] for value in polylines_dict.values() if not isinstance(value, int)]
        label, vec_glb_idx = vec_tag.split('_')
        label = int(label)
        vec_glb_idx = int(vec_glb_idx)

        if label == 0: # ped_crossing
            color = 'b'
        elif label == 1: # divider
            color = 'r'
        elif label == 2: # boundary
            color = 'g'

        if label == 0: # crossing, merged by convex hull
            vector = np.array(polylines[0]) 

            pts = vector[:, :2]
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=0.5)
            ax.plot(x, y, "o", color=color, markersize=50)
            # 在中间位置绘制tag
            mid_idx = len(pts) // 2
            ax.text(x.mean(),y.mean(), vec_tag, fontsize=12*5, ha='center', va='center', color='b')

        elif label == 1: # divider, merged fitting a polyline
            polylines_vecs = [np.array(line) for line in polylines]

            for split_idx,one_line in enumerate(polylines_vecs):
                # one_line = np.array(LineString(one_line).simplify(args.simplify*2).coords)
                one_line = np.array(one_line)
                pts = one_line[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=0.5)
                ax.plot(x, y, "o", color=color, markersize=50)
                # 在中间位置绘制tag
                mid_idx = len(pts) // 2
                ax.text(pts[mid_idx][0], pts[mid_idx][1], vec_tag+'_'+str(split_idx), fontsize=12*5, ha='center', va='center', color='black')

        elif label == 2: # boundary, do not merge
            polylines_vecs = [np.array(line) for line in polylines]

            for split_idx,one_line in enumerate(polylines_vecs):
                # one_line = np.array(LineString(one_line).simplify(args.simplify).coords)
                one_line = np.array(one_line)
                pts = one_line[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=0.5)
                ax.plot(x, y, "o", color=color, markersize=50)
                # 在中间位置绘制tag
                mid_idx = len(pts) // 2
                ax.text(pts[mid_idx][0], pts[mid_idx][1], vec_tag+'_'+str(split_idx), fontsize=12*5, ha='center', va='center', color='black')
    # 在中间位置绘制一个圆
    ax.plot(x_mid, y_mid, "o", color='black', markersize=50)
    pred_save_path = pred_save_folder +f'/{dataset}'+ f'/{frame_timestep}_{tile_name}.png'
    plt.grid(False)
    plt.savefig(pred_save_path, bbox_inches='tight', transparent=False, dpi=20)
    plt.clf() 
    plt.close(fig)
    print("global_intance_bank  : ", pred_save_path)



def plot_bank_list_distence_intervel10(bank_list,pred_save_folder,frame_timestep,x_min,y_min,x_max,y_max,tile_name,x_mid,y_mid,dataset):
    rank, world_size = get_dist_info()
    if rank != 0:
        return
    os.makedirs(pred_save_folder+f'/{dataset}', exist_ok=True)
    plt.figure(facecolor='lightgreen')
    fig = plt.figure(figsize=(int(x_max - x_min + 10) ,int(y_max-y_min + 10)))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(float(x_min), float(x_max))
    ax.set_ylim(float(y_min), float(y_max))

    for vec_tag, polylines_dict in bank_list.items():
        polylines = [value[0] for value in polylines_dict.values() if not isinstance(value, int)]
        label, vec_glb_idx = vec_tag.split('_')
        label = int(label)
        vec_glb_idx = int(vec_glb_idx)

        if label == 0: # ped_crossing
            color = 'b'
        elif label == 1: # divider
            color = 'r'
        elif label == 2: # boundary
            color = 'g'

        if label == 0: # crossing, merged by convex hull
            vector = np.array(polylines[0]) 

            pts = vector[:, :2]
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=0.5)
            ax.plot(x, y, "o", color=color, markersize=50)
            # 在中间位置绘制tag
            mid_idx = len(pts) // 2
            # ax.text(x.mean(),y.mean(), vec_tag, fontsize=12*5, ha='center', va='center', color='b')

        elif label == 1: # divider, merged fitting a polyline
            keep_numbers = len(polylines)//5 if len(polylines)>10 else 1
            polylines_vecs = [np.array(line) for line in polylines[::keep_numbers]]

            for split_idx,one_line in enumerate(polylines_vecs):
                # one_line = np.array(LineString(one_line).simplify(args.simplify*2).coords)
                one_line = np.array(one_line)
                pts = one_line[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=0.5)
                ax.plot(x, y, "o", color=color, markersize=50)
                # 在中间位置绘制tag
                mid_idx = len(pts) // 2
                # ax.text(pts[mid_idx][0], pts[mid_idx][1], vec_tag+'_'+str(split_idx), fontsize=12*5, ha='center', va='center', color='black')

        elif label == 2: # boundary, do not merge
            keep_numbers = len(polylines)//5 if len(polylines)>10 else 1
            polylines_vecs = [np.array(line) for line in polylines[::keep_numbers]]

            for split_idx,one_line in enumerate(polylines_vecs):
                # one_line = np.array(LineString(one_line).simplify(args.simplify).coords)
                one_line = np.array(one_line)
                pts = one_line[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=0.5)
                ax.plot(x, y, "o", color=color, markersize=50)
                # 在中间位置绘制tag
                mid_idx = len(pts) // 2
                # ax.text(pts[mid_idx][0], pts[mid_idx][1], vec_tag+'_'+str(split_idx), fontsize=12*5, ha='center', va='center', color='black')
    # 在中间位置绘制一个圆
    ax.plot(x_mid, y_mid, "o", color='black', markersize=50)
    pred_save_path = pred_save_folder +f'/{dataset}'+ f'/{frame_timestep}_{tile_name}.png'
    plt.grid(False)
    plt.savefig(pred_save_path, bbox_inches='tight', transparent=False, dpi=20)
    plt.clf() 
    plt.close(fig)
    print("global_intance_bank  : ", pred_save_path)


def update_global_instance_bank(global_instance_bank,instance_bank,x_min,y_min,x_max,y_max,update_global_instance_flag,bank_list,scene_matcing=True,plot_index=0):

    #instance_bank也就是单个scene里面的匹配
    if len(instance_bank) and scene_matcing==True:
        instance_bank_masks,instance_bank_masks_id = instance_bank2mask(instance_bank,x_min,y_min,x_max,y_max)
        #scene和scene的instance bank进行匹配和合并
        matchings_instance_bank= find_matchings(instance_bank_masks,instance_bank_masks, thresh=0.01,scene_matcing=True)
        # if 0 in matchings_instance_bank and   len(matchings_instance_bank[0][0]) > 1:
        #     print('DEBUG')

        for label, matching  in matchings_instance_bank.items():
            pre_matching,cur_matching = matching
            for i in range(len(pre_matching)):
                if pre_matching[i] != -1:
                    pre_matching[pre_matching[i]] = -1#防止遍历两次,因为是同一个instance_bank的元素
                    pre_tag = instance_bank_masks_id[label][i]
                    cur_tag = instance_bank_masks_id[label][pre_matching[i]]
                    if len(instance_bank[cur_tag]) == 0:
                        instance_bank.pop(cur_tag)
                        continue
                    polylines = instance_bank[pre_tag]+instance_bank[cur_tag]
                    print(pre_tag,cur_tag)
                    if label == 0: # crossing, merged by convex hull
                        polygon = merge_corssing(polylines)
                        polygon = polygon.simplify(0.5)
                        vector = np.array(polygon.exterior.coords) 
                        updated_polyline = LineString(vector)
                        instance_bank[cur_tag] = [updated_polyline, ]
                    elif label == 1:
                        polylines_vecs = [np.array(one_line.coords) for one_line in polylines]
                        polylines_vecs = merge_divider(polylines_vecs)
                        updated_polylines = [LineString(vec) for vec in polylines_vecs]
                        instance_bank[cur_tag] = updated_polylines
                    else:
                        polylines_vecs = [np.array(one_line.coords) for one_line in polylines]
                        polylines_vecs = merge_boundary(polylines_vecs)
                        updated_polylines = [LineString(vec) for vec in polylines_vecs]
                        instance_bank[cur_tag] = updated_polylines
                    instance_bank.pop(pre_tag)

        draw_instance_bank2mask(instance_bank_masks,f'{plot_index}')

    if len(global_instance_bank) and len(instance_bank):
        #合并instance bank和global_instance_bank,合并后的放到global_instance_bank中
        global_instance_bank_masks,global_instance_bank_masks_id = instance_bank2mask(global_instance_bank,x_min,y_min,x_max,y_max)
        instance_bank_masks,instance_bank_masks_id = instance_bank2mask(instance_bank,x_min,y_min,x_max,y_max)


        #global和scene的instance bank进行匹配和合并
        matchings= find_matchings(instance_bank_masks,global_instance_bank_masks, thresh=0.01)

        for label, matching  in matchings.items():
            pre_matching,cur_matching = matching
            for i in range(len(pre_matching)):
                if pre_matching[i] != -1:
                    pre_tag = instance_bank_masks_id[label][i]
                    cur_tag = global_instance_bank_masks_id[label][pre_matching[i]]
                    polylines = instance_bank[pre_tag]+global_instance_bank[cur_tag]
                    print(pre_tag,cur_tag)
                    if label == 0: # crossing, merged by convex hull
                        polygon = merge_corssing(polylines)
                        polygon = polygon.simplify(0.5)
                        vector = np.array(polygon.exterior.coords) 
                        updated_polyline = LineString(vector)
                        global_instance_bank[cur_tag] = [updated_polyline, ]
                    elif label == 1:
                        polylines_vecs = [np.array(one_line.coords) for one_line in polylines]
                        polylines_vecs = merge_divider(polylines_vecs)
                        updated_polylines = [LineString(vec) for vec in polylines_vecs]
                        global_instance_bank[cur_tag] = updated_polylines
                    else:
                        polylines_vecs = [np.array(one_line.coords) for one_line in polylines]
                        polylines_vecs = merge_boundary(polylines_vecs)
                        updated_polylines = [LineString(vec) for vec in polylines_vecs]
                        global_instance_bank[cur_tag] = updated_polylines
                    instance_bank.pop(pre_tag)

                    #处理bank_list
                    pre2cur_vectors = {}
                    for x_y in set(list(bank_list[pre_tag].keys())+list(bank_list[cur_tag].keys())):
                        if x_y in bank_list[pre_tag] and x_y in bank_list[cur_tag]:
                            if bank_list[pre_tag][x_y][1] >= bank_list[cur_tag][x_y][1]:
                                pre2cur_vectors[x_y] = bank_list[pre_tag][x_y]
                            else:
                                pre2cur_vectors[x_y] = bank_list[cur_tag][x_y]
                        else:
                            if x_y in bank_list[pre_tag]:
                                pre2cur_vectors[x_y] = bank_list[pre_tag][x_y]
                            else:
                                pre2cur_vectors[x_y] = bank_list[cur_tag][x_y]
                            # pre2cur_vectors = bank_list[pre_tag][x_y]+bank_list[cur_tag][x_y]
                            # bank_list[cur_tag][x_y] = pre2cur_vectors
                            # bank_list.pop(pre_tag)
                    # pre2cur_vectors = bank_list[pre_tag]+bank_list[cur_tag]
                    bank_list[cur_tag] = pre2cur_vectors
                    bank_list.pop(pre_tag)

        
    if update_global_instance_flag:
        global_instance_bank.update(instance_bank)
        instance_bank.clear()



def get_prev2curr_vectors(vecs=None, prev2curr_matrix=None,origin=None,roi_size=None, denormalize=False, clip=False,result_normalize=False):
    # transform prev vectors
    if len(vecs) > 0:
        vecs = np.stack(vecs, 0)
        vecs = torch.tensor(vecs)
        N, num_points, _ = vecs.shape
        if denormalize:
            denormed_vecs = vecs * np.array(roi_size) + np.array(origin) # (num_prop, num_pts, 2)
        else:
            denormed_vecs = vecs
        z = denormed_vecs.new_zeros((N, num_points, 1))
        z[:] = -1.6
        denormed_vecs = torch.cat([
            denormed_vecs,
            z, # z-axis
            denormed_vecs.new_ones((N, num_points, 1)) # 4-th dim
        ], dim=-1) # (num_prop, num_pts, 4)

        transformed_vecs = torch.einsum('lk,ijk->ijl', prev2curr_matrix, denormed_vecs.double()).float()
        vecs = transformed_vecs[..., :2]
        if result_normalize:
            vecs = (transformed_vecs[..., :2] - torch.tensor(origin)) /  torch.tensor(roi_size) # (num_prop, num_pts, 2)
        if clip:
            vecs = torch.clip(vecs, min=0., max=1.)
        # vecs = vecs * roi_size + origin
    
    return vecs


def get_patch_coord(patch_box: Tuple[float, float, float, float],
                    patch_angle: float = 0.0) -> Polygon:
    """
    Convert patch_box to shapely Polygon coordinates.
    :param patch_box: Patch box defined as [x_center, y_center, height, width].
    :param patch_angle: Patch orientation in degrees.
    :return: Box Polygon for patch_box.
    """
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

    return patch