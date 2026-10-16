USE_TRACK_QUERY=True
_base_ = [
    '../../_base_/default_runtime.py'
]
DEBUG = False
# DEBUG = False
SMALL_BATCH = False if not DEBUG else True
# SMALL_BATCH = True
TILE=False
SMALL_BATCH_DEBUG=False if not DEBUG else True
PRIOR_MODE = 'v1'#v1:不加先验;v2:加上一帧先验;v3:存储所有大于阈值的bank_list,加最近距离先验(新方案);v4:在v3的基础上,加memory_bank cross_attn
use_memory_bank_new=True if PRIOR_MODE == 'v4' else False

# if DEBUG:
#     data_path_ann_file = './datasets/nuscenes/nuscenes_map_infos_val_scene-0003.pkl'
#     if TILE:
#         data_path_ann_file = './datasets/nuscenes/nuscenes_map_infos_val_tile.pkl'

# 统计 GPU 卡的数量
gpu_count = len( __import__('os').popen('nvidia-smi -L').read().strip().split('\n'))
gpu_info = str(__import__('os').popen('nvidia-smi --query-gpu=name --format=csv,noheader').read().strip())

# model type
type = 'Mapper'
plugin = True

# plugin code dir
plugin_dir = 'plugin/'
[]
# img configs
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

img_h = 480
img_w = 800
img_size = (img_h, img_w)

num_gpus = 8
if 'A100' in gpu_info:
    batch_size = 3
    gpu_info = 'A100'
elif 'A800' in gpu_info:
    batch_size = 14
    gpu_info = 'A800'
elif 'V100' in gpu_info:
    batch_size = 3
    gpu_info = 'V100'
else:
    batch_size = 2
    gpu_info = '3090'

print(f'DEBUG: {DEBUG}', f'SMALL_BATCH: {SMALL_BATCH}',f'GPU_COUNT: {gpu_count}', f'GPU_INFO: {gpu_info}', f'BATCH_SIZE: {batch_size}',f'TILE: {TILE}')

map_attribute = {
    'root_dir': './work_dirs/map_slices/raster_global_map',
    'type': 'rasterized',
    'prefix': 'map_large_reso_gru_cpu',
    'tile_param': {
        'data_type': 'float32',
        'embed_dims': 256//4,
        'num_traversals': 1, },
    'batch_size': num_gpus if not DEBUG else 1,#这里的意思是有几个GPU,如果是本地DEBUG且模拟集群某个rank的异常情况,这里选择8
    'single_gpu': False,
    'global_map_tile_size': [4, 4],
    'global_map_raster_size': [0.30, 0.30],
}

sort_train_infos = True
sort_val_infos = True
sort_batch_size = map_attribute['batch_size']





num_iters_per_epoch = 27968 // (num_gpus * batch_size)
num_epochs = 48
num_epochs_interval = num_epochs // 8
total_iters = num_epochs * num_iters_per_epoch
num_queries = 100

# category configs
cat2id = {
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
}
num_class = max(list(cat2id.values())) + 1

# bev configs
roi_size = (60, 30) # bev range, 60m in x-axis, 30m in y-axis
bev_h = 50
bev_w = 100
pc_range = [-roi_size[0]/2, -roi_size[1]/2, -3, roi_size[0]/2, roi_size[1]/2, 5]

# vectorize params
coords_dim = 2
sample_dist = -1
sample_num = -1
simplify = True

# rasterize params (for temporal matching use)
canvas_size = (200, 100) # bev feature size
thickness = 3 # thickness of rasterized polylines

# meta info for submission pkl
meta = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
    output_format='vector')

# model configs
bev_embed_dims = 256
embed_dims = 512
num_feat_levels = 3
norm_cfg = dict(type='BN2d')
num_class = max(list(cat2id.values()))+1
num_points = 20
permute = True
use_prev_pred=True
model = dict(
    type='MapTracker',
    PRIOR_MODE=PRIOR_MODE,
    roi_size=roi_size,
    bev_h=bev_h,
    bev_w=bev_w,
    history_steps=1,#trains时,最多存储多少帧存储历史bev_feature
    test_time_history_steps=1,#test时:1.use_memory中的memory_bank:代表某个track_query最多存储bank_size次出现的embedding2.存储多少帧历史bev_feature
    mem_select_dist_ranges=[1, 5, 10, 15],
    skip_vector_head=False,
    freeze_bev=False,
    track_fp_aug=False,
    use_memory=False,
    mem_len=4,#train-use_memory中的memory_bank:代表某个track_query最多存储bank_size次出现的embedding
    mem_warmup_iters=-1,
    use_prev_pred=use_prev_pred,#使用坐标,类别,置信度编码为track_query以及vector_buffer
    use_prev_pred_encode='v1',#'v1'是prev_pred的坐标,类别,置信度, 'v2'是mapex硬编码
    add_noise_big_ratio=False,#'v1'是直接在原本的track_query和vector_buffer上加大噪声, 'v2'是trans_loss加一个辅助分支的学习大噪声proppogate
    map_attribute=map_attribute,
    open_nmp=True,
    use_memory_bank_new=use_memory_bank_new,
    backbone_cfg=dict(
        type='BEVFormerBackbone',
        roi_size=roi_size,
        bev_h=bev_h,
        bev_w=bev_w,
        history_steps=4,#paddding,不够就会补为0的bev_feature,不用改,直接让模型学到无效的padding就行
        use_grid_mask=True,
        img_backbone=dict(
            type='ResNet',
            with_cp=False,
            # pretrained='./resnet50_checkpoint.pth',
            pretrained='open-mmlab://detectron2/resnet50_caffe',
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=-1,
            norm_cfg=norm_cfg,
            norm_eval=True,
            style='caffe',
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True)
            ),
        img_neck=dict(
            type='FPN',
            in_channels=[512, 1024, 2048],
            out_channels=bev_embed_dims,
            start_level=0,
            add_extra_convs=True,
            num_outs=num_feat_levels,
            norm_cfg=norm_cfg,
            relu_before_extra_convs=True),
        transformer=dict(
            type='PerceptionTransformer',
            embed_dims=bev_embed_dims,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=2,
                pc_range=pc_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=bev_embed_dims,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=bev_embed_dims,
                                num_points=8,
                                num_levels=num_feat_levels),
                            embed_dims=bev_embed_dims,
                        )
                    ],
                    feedforward_channels=bev_embed_dims*2,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                    'ffn', 'norm')
                )
            ),
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=bev_embed_dims//2,
            row_num_embed=bev_h,
            col_num_embed=bev_w,
            ),
    ),
    head_cfg=dict(
        type='MapDetectorHead',
        num_queries=num_queries,
        embed_dims=embed_dims,
        num_classes=num_class,
        in_channels=bev_embed_dims,
        num_points=num_points,
        roi_size=roi_size,
        coord_dim=2,
        different_heads=False,
        predict_refine=False,
        sync_cls_avg_factor=True,
        trans_loss_weight=0.1*100,
        use_prev_pred=use_prev_pred,
        transformer=dict(
            type='MapTransformer',
            num_feature_levels=1,
            num_points=num_points,
            coord_dim=2,
            encoder=dict(
                type='PlaceHolderEncoder',
                embed_dims=embed_dims,
            ),
            decoder=dict(
                type='MapTransformerDecoder_new',
                num_layers=6,
                prop_add_stage=1,
                return_intermediate=True,
                transformerlayers=dict(
                    type='MapTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            attn_drop=0.1,
                            proj_drop=0.1,
                        ),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            num_levels=1,
                            num_points=num_points,
                            dropout=0.1,
                        ),
                        dict(
                            type='MultiheadAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            attn_drop=0.1,
                            proj_drop=0.1,
                        ),
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=embed_dims,
                        feedforward_channels=embed_dims*2,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),        
                    ),
                    feedforward_channels=embed_dims*2,
                    ffn_dropout=0.1,
                    ## an addtional cross attention for vector memory fusion
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'cross_attn', 'norm',
                                    'ffn', 'norm')
                )
            )
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5.0
        ),
        loss_reg=dict(
            type='LinesL1Loss',
            loss_weight=50.0,
            beta=0.01,
        ),
        assigner=dict(
            type='HungarianLinesAssigner',
                cost=dict(
                    type='MapQueriesCost',
                    cls_cost=dict(type='FocalLossCost', weight=5.0),
                    reg_cost=dict(type='LinesL1Cost', weight=50.0, beta=0.01, permute=permute),
                    ),
                ),
        ),
    seg_cfg=dict(
        type='MapSegHead',
        num_classes=num_class,
        in_channels=bev_embed_dims,
        embed_dims=bev_embed_dims,
        bev_size=(bev_w, bev_h),
        canvas_size=canvas_size,
        loss_seg=dict(
            type='MaskFocalLoss',
            use_sigmoid=True,
            loss_weight=10.0,
        ),
        loss_dice=dict(
            type='MaskDiceLoss',
            loss_weight=1.0,
        )
    ),
    model_name='SingleStage'
)

# data processing pipelines
train_pipeline = [
    dict(
        type='VectorizeMap',
        coords_dim=coords_dim,
        roi_size=roi_size,
        sample_num=num_points,
        normalize=True,
        permute=permute,
    ),
    dict(
        type='RasterizeMap',   
        roi_size=roi_size,
        coords_dim=coords_dim,
        canvas_size=canvas_size,
        thickness=thickness,
        semantic_mask=True,
    ),
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='ResizeMultiViewImages',
         size=img_size, # H, W
         change_intrinsics=True,
         ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img', 'vectors', 'semantic_mask'], meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'))
]

# data processing pipelines
test_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='ResizeMultiViewImages',
         size=img_size, # H, W
         change_intrinsics=True,
         ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img'], meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'))
]




# configs for evaluation code
# DO NOT CHANGE
eval_config = dict(
    type='NuscDataset',
    data_root='./datasets/nuscenes',
    ann_file='./datasets/nuscenes/nuscenes_map_infos_val.pkl',
    meta=meta,
    roi_size=roi_size,
    cat2id=cat2id,
    pipeline=[
        dict(
            type='VectorizeMap',
            coords_dim=coords_dim,
            simplify=True,
            normalize=False,
            roi_size=roi_size
        ),
        dict(type='FormatBundleMap'),
        dict(type='Collect3D', keys=['vectors',], meta_keys=['token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'])
    ],
    interval=1,
    sort_train_infos=sort_train_infos,
    sort_val_infos=sort_val_infos,
    sort_batch_size=sort_batch_size,
    matching=True,
    gm_grid_size=map_attribute['global_map_tile_size'],

)


match_config = dict(
    type='NuscDataset',
    data_root='./datasets/nuscenes',
    ann_file='./datasets/nuscenes/nuscenes_map_infos_val.pkl' ,
    meta=meta,
    roi_size=roi_size,
    cat2id=cat2id,
    pipeline=[
        dict(
            type='VectorizeMap',
            coords_dim=coords_dim,
            simplify=False,
            normalize=True,
            roi_size=roi_size,
            sample_num=num_points,
        ),
        dict(
            type='RasterizeMap',   
            roi_size=roi_size,
            coords_dim=coords_dim,
            canvas_size=canvas_size,
            thickness=thickness,
        ),
        dict(type='FormatBundleMap'),
        dict(type='Collect3D', keys=['vectors', 'semantic_mask'], meta_keys=['token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'])
    ],
    interval=1,

    sort_train_infos=sort_train_infos,
    sort_val_infos=sort_val_infos,
    sort_batch_size=sort_batch_size,
    gm_grid_size=map_attribute['global_map_tile_size'],
    nusc_city_split=False,
    matching=True,
)

# dataset configs
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=8 if not DEBUG else 0,
    train=dict(
        type='NuscDataset',
        data_root='./datasets/nuscenes',
        ann_file='./datasets/nuscenes/nuscenes_map_infos_train.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=train_pipeline,
        seq_split_num=-2,
        matching=True,
        multi_frame=0,#5包括当前帧,这里的作用是选择多少-1之前帧
        sampling_span=10,#在前10帧里面随机选择multi_frame帧
        
        sort_train_infos=sort_train_infos,
        sort_val_infos=sort_val_infos,
        sort_batch_size=sort_batch_size,
        gm_grid_size=map_attribute['global_map_tile_size'],
        nusc_city_split=False,
    ),
    val=dict(
        type='NuscDataset',
        data_root='./datasets/nuscenes',
        ann_file='./datasets/nuscenes/nuscenes_map_infos_val.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=1,
        matching=True,
        sort_train_infos=sort_train_infos,
        sort_val_infos=sort_val_infos,
        sort_batch_size=sort_batch_size,
        gm_grid_size=map_attribute['global_map_tile_size'],
        nusc_city_split=False,
    ),
    test=dict(
        type='NuscDataset',
        data_root='./datasets/nuscenes',
        ann_file='./datasets/nuscenes/nuscenes_map_infos_val.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=1,
        matching=True,
        sort_train_infos=sort_train_infos,
        sort_val_infos=sort_val_infos,
        sort_batch_size=sort_batch_size,
        gm_grid_size=map_attribute['global_map_tile_size'],
        nusc_city_split=False,
    ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
    # shuffle=True,
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=5e-4 * 0.01,
    paramwise_cfg=dict(
        custom_keys={
            'backbone.img_backbone': dict(lr_mult=0.1),
            'backbone.img_neck': dict(lr_mult=0.1),
            'backbone.transformer': dict(lr_mult=0.5),
            'backbone.positional_encoding': dict(lr_mult=0.5),
            'seg_decoder': dict(lr_mult=0.5),
            #额外新增一个
            # 'backbone.transformer.encoder': dict(lr_mult=0.1),

        }),
    weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy & schedule
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500 if not DEBUG else 1000000,
    # warmup_iters=5000 if not DEBUG else 1000000,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=3e-3)

evaluation = dict(interval=num_epochs_interval*num_iters_per_epoch) if not SMALL_BATCH else dict(interval=20)
#evaluation = dict(interval=1) # for debugging use..
find_unused_parameters = True #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=num_epochs_interval*num_iters_per_epoch) if not SMALL_BATCH else dict(interval=500000)

runner = dict(
    type='MyRunnerWrapper', max_iters=num_epochs * num_iters_per_epoch)

custom_hooks = [dict(type='SetEpochInfoHook1', priority='VERY_HIGH', interval=num_iters_per_epoch if not DEBUG else 100/batch_size)]

log_config = dict(
    interval=50 if not SMALL_BATCH else 10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

SyncBN = True
# 三阶段保存的pth
# load_from = 'pretrained_ckpts/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune/latest.pth'
# if DEBUG:
#     # load_from = "work_dirs/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune/iter_2500.pth"
#     load_from = "pretrained_ckpts/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup/latest.pth"
# else:


# if use_memory_bank_new:
#     #时序+cross_attn的pth
#     load_from = "pretrained_ckpts/mapTracker54/latest.pth"
# else:
#     #时序的pth
#     load_from = "pretrained_ckpts/mapTracker29-exp3/latest.pth"
#实验4训练完的pth
# load_from = "train_reslut_pth/iter_24444.pth"

load_from = "pretrained_ckpts/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup/latest.pth"