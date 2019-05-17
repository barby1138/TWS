#!/bin/bash

cd ../maskrcnn-benchmark/maskrcnn_benchmark

# CS
#python ../tools/train_net.py --config-file "../configs/ds_eval/e2e_mask_rcnn_R_50_FPN_1x_cocostyle.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 96000 SOLVER.STEPS "(48000, 64000)" TEST.IMS_PER_BATCH 1 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000
#python ../tools/test_net.py --config-file "../configs/ds_eval/e2e_mask_rcnn_R_50_FPN_1x_cocostyle.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 96000 SOLVER.STEPS "(48000, 64000)" TEST.IMS_PER_BATCH 1 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000

# KITTI
#python ../tools/train_net.py --config-file "../configs/ds_eval/e2e_mask_rcnn_R_50_FPN_1x_cocostyle_kitti.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 45000 SOLVER.STEPS "(48000, 64000)" TEST.IMS_PER_BATCH 1 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000
python ../tools/test_net.py --config-file "../configs/ds_eval/e2e_mask_rcnn_R_50_FPN_1x_cocostyle_kitti.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 96000 SOLVER.STEPS "(48000, 64000)" TEST.IMS_PER_BATCH 1 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000

# VKITTI-clone
#python ../tools/train_net.py --config-file "../configs/ds_eval/e2e_mask_rcnn_R_50_FPN_1x_cocostyle_vkitti_clone.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 96000 SOLVER.STEPS "(48000, 64000)" TEST.IMS_PER_BATCH 1 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000
#python ../tools/test_net.py --config-file "../configs/ds_eval/e2e_mask_rcnn_R_50_FPN_1x_cocostyle_vkitti_clone.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 96000 SOLVER.STEPS "(48000, 64000)" TEST.IMS_PER_BATCH 1 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000

# TWS
#python ../tools/train_net.py --config-file "../configs/ds_eval/e2e_mask_rcnn_R_50_FPN_1x_cocostyle_tws.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 96000 SOLVER.STEPS "(48000, 64000)" TEST.IMS_PER_BATCH 1 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000
#python ../tools/test_net.py --config-file "../configs/ds_eval/e2e_mask_rcnn_R_50_FPN_1x_cocostyle_tws.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 96000 SOLVER.STEPS "(48000, 64000)" TEST.IMS_PER_BATCH 1 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000
