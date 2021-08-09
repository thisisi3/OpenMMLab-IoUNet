_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model=dict(
    roi_head=dict(
        type='IoURoIHead',
        bbox_roi_extractor=dict(
            _delete_=True,
            type='PrRoIExtractor',
            roi_layer=dict(type='PrRoIPool2D', pooled_height=7, pooled_width=7),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        roi_generator=dict(
            pre_sample=4000,
            xy_steps=16,
            wh_steps=16,
            xy_range=(0, 1),
            area_range=(1/3, 3),
            nonlinearity=2,
            per_iou=None,
            sample_num=1000,
            max_num=1000,
            compensate=None),
        iou_head=dict(
            type='IoUHead',
            in_channels=256 * 7 * 7,
            fc_channels=[1024, 1024],
            num_classes=80,
            class_agnostic=False,
            target_norm=dict(mean=0.5, std=0.5),
            loss_iou=dict(type='SmoothL1Loss', loss_weight=5.0))),
    train_cfg=dict(
        rcnn=dict(
            _delete_=True,
            bbox_assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            bbox_sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            iou_assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            iou_sampler=dict(type='PseudoSampler'),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rcnn=dict(
            iou=dict(
                nms=dict(multiclass=True, iou_threshold=0.5),
                refine=dict(
                    pre_refine=100,
                    t=5,
                    omega_1=0.001,
                    omega_2=-0.01,
                    lamb=0.5,
                    use_iou_score=True)))))
