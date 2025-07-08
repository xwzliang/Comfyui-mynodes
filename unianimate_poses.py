import os
import json
import importlib.util
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from comfy.utils import ProgressBar
import comfy.model_management as mm
# workspace/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/unianimate/nodes.py
# 1) point at the .py file
file_path = Path("custom_nodes/ComfyUI-WanVideoWrapper/unianimate/nodes.py").resolve()

# 2) choose a module name that *matches* its location
mod_name = "custom_nodes.ComfyUI-WanVideoWrapper.unianimate.nodes"
pkg_name = "custom_nodes.ComfyUI-WanVideoWrapper.unianimate"

spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
module = importlib.util.module_from_spec(spec)

# 3) tell Python which package this module belongs to, so relative imports work
module.__package__ = pkg_name

# 4) execute (this will allow `from ..utils import log` to resolve)
spec.loader.exec_module(module)

# 5) (optional) register it in sys.modules so further imports can find it
sys.modules[mod_name] = module

def my_pose_extract(pose_images, ref_image, dwpose_model, height, width, score_threshold, stick_width,
                 draw_body=True, draw_hands=True, hand_keypoint_size=4, draw_feet=True,
                 body_keypoint_size=4, handle_not_detected="repeat", draw_head=True, movement_thresh=0.0):
    results_vis = []
    comfy_pbar = ProgressBar(len(pose_images))

    if ref_image is not None:
        try:
            pose_ref = dwpose_model(ref_image.squeeze(0), score_threshold=score_threshold)
        except:
            raise ValueError("No pose detected in reference image")
    prev_pose = None
    for img in tqdm(pose_images, desc="Pose Extraction", unit="image", total=len(pose_images)):
        try:
            pose = dwpose_model(img, score_threshold=score_threshold)
            if handle_not_detected == "repeat":
                prev_pose = pose
        except:
            if prev_pose is not None:
                pose = prev_pose
            else:
                pose = np.zeros_like(img)
        results_vis.append(pose)
        comfy_pbar.update(1)
    
    # 3) make a working copy where we'll apply transforms
    results_vis_copy = deepcopy(results_vis)
    img_diag = np.hypot(width, height)
    
    bodies = results_vis[0]['bodies']
    faces = results_vis[0]['faces']
    hands = results_vis[0]['hands']
    candidate = bodies['candidate']

    if ref_image is not None:
        ref_bodies = pose_ref['bodies']
        ref_faces = pose_ref['faces']
        ref_hands = pose_ref['hands']
        ref_candidate = ref_bodies['candidate']


        ref_2_x = ref_candidate[2][0]
        ref_2_y = ref_candidate[2][1]
        ref_5_x = ref_candidate[5][0]
        ref_5_y = ref_candidate[5][1]
        ref_8_x = ref_candidate[8][0]
        ref_8_y = ref_candidate[8][1]
        ref_11_x = ref_candidate[11][0]
        ref_11_y = ref_candidate[11][1]
        ref_center1 = 0.5*(ref_candidate[2]+ref_candidate[5])
        ref_center2 = 0.5*(ref_candidate[8]+ref_candidate[11])

        zero_2_x = candidate[2][0]
        zero_2_y = candidate[2][1]
        zero_5_x = candidate[5][0]
        zero_5_y = candidate[5][1]
        zero_8_x = candidate[8][0]
        zero_8_y = candidate[8][1]
        zero_11_x = candidate[11][0]
        zero_11_y = candidate[11][1]
        zero_center1 = 0.5*(candidate[2]+candidate[5])
        zero_center2 = 0.5*(candidate[8]+candidate[11])

        x_ratio = (ref_5_x-ref_2_x)/(zero_5_x-zero_2_x)
        y_ratio = (ref_center2[1]-ref_center1[1])/(zero_center2[1]-zero_center1[1])

        results_vis[0]['bodies']['candidate'][:,0] *= x_ratio
        results_vis[0]['bodies']['candidate'][:,1] *= y_ratio
        results_vis[0]['faces'][:,:,0] *= x_ratio
        results_vis[0]['faces'][:,:,1] *= y_ratio
        results_vis[0]['hands'][:,:,0] *= x_ratio
        results_vis[0]['hands'][:,:,1] *= y_ratio
        
        ########neck########
        l_neck_ref = ((ref_candidate[0][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[0][1] - ref_candidate[1][1]) ** 2) ** 0.5
        l_neck_0 = ((candidate[0][0] - candidate[1][0]) ** 2 + (candidate[0][1] - candidate[1][1]) ** 2) ** 0.5
        neck_ratio = l_neck_ref / l_neck_0

        x_offset_neck = (candidate[1][0]-candidate[0][0])*(1.-neck_ratio)
        y_offset_neck = (candidate[1][1]-candidate[0][1])*(1.-neck_ratio)

        results_vis[0]['bodies']['candidate'][0,0] += x_offset_neck
        results_vis[0]['bodies']['candidate'][0,1] += y_offset_neck
        results_vis[0]['bodies']['candidate'][14,0] += x_offset_neck
        results_vis[0]['bodies']['candidate'][14,1] += y_offset_neck
        results_vis[0]['bodies']['candidate'][15,0] += x_offset_neck
        results_vis[0]['bodies']['candidate'][15,1] += y_offset_neck
        results_vis[0]['bodies']['candidate'][16,0] += x_offset_neck
        results_vis[0]['bodies']['candidate'][16,1] += y_offset_neck
        results_vis[0]['bodies']['candidate'][17,0] += x_offset_neck
        results_vis[0]['bodies']['candidate'][17,1] += y_offset_neck
        
        ########shoulder2########
        l_shoulder2_ref = ((ref_candidate[2][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[2][1] - ref_candidate[1][1]) ** 2) ** 0.5
        l_shoulder2_0 = ((candidate[2][0] - candidate[1][0]) ** 2 + (candidate[2][1] - candidate[1][1]) ** 2) ** 0.5

        shoulder2_ratio = l_shoulder2_ref / l_shoulder2_0

        x_offset_shoulder2 = (candidate[1][0]-candidate[2][0])*(1.-shoulder2_ratio)
        y_offset_shoulder2 = (candidate[1][1]-candidate[2][1])*(1.-shoulder2_ratio)

        results_vis[0]['bodies']['candidate'][2,0] += x_offset_shoulder2
        results_vis[0]['bodies']['candidate'][2,1] += y_offset_shoulder2
        results_vis[0]['bodies']['candidate'][3,0] += x_offset_shoulder2
        results_vis[0]['bodies']['candidate'][3,1] += y_offset_shoulder2
        results_vis[0]['bodies']['candidate'][4,0] += x_offset_shoulder2
        results_vis[0]['bodies']['candidate'][4,1] += y_offset_shoulder2
        results_vis[0]['hands'][1,:,0] += x_offset_shoulder2
        results_vis[0]['hands'][1,:,1] += y_offset_shoulder2

        ########shoulder5########
        l_shoulder5_ref = ((ref_candidate[5][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[5][1] - ref_candidate[1][1]) ** 2) ** 0.5
        l_shoulder5_0 = ((candidate[5][0] - candidate[1][0]) ** 2 + (candidate[5][1] - candidate[1][1]) ** 2) ** 0.5

        shoulder5_ratio = l_shoulder5_ref / l_shoulder5_0

        x_offset_shoulder5 = (candidate[1][0]-candidate[5][0])*(1.-shoulder5_ratio)
        y_offset_shoulder5 = (candidate[1][1]-candidate[5][1])*(1.-shoulder5_ratio)

        results_vis[0]['bodies']['candidate'][5,0] += x_offset_shoulder5
        results_vis[0]['bodies']['candidate'][5,1] += y_offset_shoulder5
        results_vis[0]['bodies']['candidate'][6,0] += x_offset_shoulder5
        results_vis[0]['bodies']['candidate'][6,1] += y_offset_shoulder5
        results_vis[0]['bodies']['candidate'][7,0] += x_offset_shoulder5
        results_vis[0]['bodies']['candidate'][7,1] += y_offset_shoulder5
        results_vis[0]['hands'][0,:,0] += x_offset_shoulder5
        results_vis[0]['hands'][0,:,1] += y_offset_shoulder5

        ########arm3########
        l_arm3_ref = ((ref_candidate[3][0] - ref_candidate[2][0]) ** 2 + (ref_candidate[3][1] - ref_candidate[2][1]) ** 2) ** 0.5
        l_arm3_0 = ((candidate[3][0] - candidate[2][0]) ** 2 + (candidate[3][1] - candidate[2][1]) ** 2) ** 0.5

        arm3_ratio = l_arm3_ref / l_arm3_0

        x_offset_arm3 = (candidate[2][0]-candidate[3][0])*(1.-arm3_ratio)
        y_offset_arm3 = (candidate[2][1]-candidate[3][1])*(1.-arm3_ratio)

        results_vis[0]['bodies']['candidate'][3,0] += x_offset_arm3
        results_vis[0]['bodies']['candidate'][3,1] += y_offset_arm3
        results_vis[0]['bodies']['candidate'][4,0] += x_offset_arm3
        results_vis[0]['bodies']['candidate'][4,1] += y_offset_arm3
        results_vis[0]['hands'][1,:,0] += x_offset_arm3
        results_vis[0]['hands'][1,:,1] += y_offset_arm3

        ########arm4########
        l_arm4_ref = ((ref_candidate[4][0] - ref_candidate[3][0]) ** 2 + (ref_candidate[4][1] - ref_candidate[3][1]) ** 2) ** 0.5
        l_arm4_0 = ((candidate[4][0] - candidate[3][0]) ** 2 + (candidate[4][1] - candidate[3][1]) ** 2) ** 0.5

        arm4_ratio = l_arm4_ref / l_arm4_0

        x_offset_arm4 = (candidate[3][0]-candidate[4][0])*(1.-arm4_ratio)
        y_offset_arm4 = (candidate[3][1]-candidate[4][1])*(1.-arm4_ratio)

        results_vis[0]['bodies']['candidate'][4,0] += x_offset_arm4
        results_vis[0]['bodies']['candidate'][4,1] += y_offset_arm4
        results_vis[0]['hands'][1,:,0] += x_offset_arm4
        results_vis[0]['hands'][1,:,1] += y_offset_arm4

        ########arm6########
        l_arm6_ref = ((ref_candidate[6][0] - ref_candidate[5][0]) ** 2 + (ref_candidate[6][1] - ref_candidate[5][1]) ** 2) ** 0.5
        l_arm6_0 = ((candidate[6][0] - candidate[5][0]) ** 2 + (candidate[6][1] - candidate[5][1]) ** 2) ** 0.5

        arm6_ratio = l_arm6_ref / l_arm6_0

        x_offset_arm6 = (candidate[5][0]-candidate[6][0])*(1.-arm6_ratio)
        y_offset_arm6 = (candidate[5][1]-candidate[6][1])*(1.-arm6_ratio)

        results_vis[0]['bodies']['candidate'][6,0] += x_offset_arm6
        results_vis[0]['bodies']['candidate'][6,1] += y_offset_arm6
        results_vis[0]['bodies']['candidate'][7,0] += x_offset_arm6
        results_vis[0]['bodies']['candidate'][7,1] += y_offset_arm6
        results_vis[0]['hands'][0,:,0] += x_offset_arm6
        results_vis[0]['hands'][0,:,1] += y_offset_arm6

        ########arm7########
        l_arm7_ref = ((ref_candidate[7][0] - ref_candidate[6][0]) ** 2 + (ref_candidate[7][1] - ref_candidate[6][1]) ** 2) ** 0.5
        l_arm7_0 = ((candidate[7][0] - candidate[6][0]) ** 2 + (candidate[7][1] - candidate[6][1]) ** 2) ** 0.5

        arm7_ratio = l_arm7_ref / l_arm7_0

        x_offset_arm7 = (candidate[6][0]-candidate[7][0])*(1.-arm7_ratio)
        y_offset_arm7 = (candidate[6][1]-candidate[7][1])*(1.-arm7_ratio)

        results_vis[0]['bodies']['candidate'][7,0] += x_offset_arm7
        results_vis[0]['bodies']['candidate'][7,1] += y_offset_arm7
        results_vis[0]['hands'][0,:,0] += x_offset_arm7
        results_vis[0]['hands'][0,:,1] += y_offset_arm7

        ########head14########
        l_head14_ref = ((ref_candidate[14][0] - ref_candidate[0][0]) ** 2 + (ref_candidate[14][1] - ref_candidate[0][1]) ** 2) ** 0.5
        l_head14_0 = ((candidate[14][0] - candidate[0][0]) ** 2 + (candidate[14][1] - candidate[0][1]) ** 2) ** 0.5

        head14_ratio = l_head14_ref / l_head14_0

        x_offset_head14 = (candidate[0][0]-candidate[14][0])*(1.-head14_ratio)
        y_offset_head14 = (candidate[0][1]-candidate[14][1])*(1.-head14_ratio)

        results_vis[0]['bodies']['candidate'][14,0] += x_offset_head14
        results_vis[0]['bodies']['candidate'][14,1] += y_offset_head14
        results_vis[0]['bodies']['candidate'][16,0] += x_offset_head14
        results_vis[0]['bodies']['candidate'][16,1] += y_offset_head14

        ########head15########
        l_head15_ref = ((ref_candidate[15][0] - ref_candidate[0][0]) ** 2 + (ref_candidate[15][1] - ref_candidate[0][1]) ** 2) ** 0.5
        l_head15_0 = ((candidate[15][0] - candidate[0][0]) ** 2 + (candidate[15][1] - candidate[0][1]) ** 2) ** 0.5

        head15_ratio = l_head15_ref / l_head15_0

        x_offset_head15 = (candidate[0][0]-candidate[15][0])*(1.-head15_ratio)
        y_offset_head15 = (candidate[0][1]-candidate[15][1])*(1.-head15_ratio)

        results_vis[0]['bodies']['candidate'][15,0] += x_offset_head15
        results_vis[0]['bodies']['candidate'][15,1] += y_offset_head15
        results_vis[0]['bodies']['candidate'][17,0] += x_offset_head15
        results_vis[0]['bodies']['candidate'][17,1] += y_offset_head15

        ########head16########
        l_head16_ref = ((ref_candidate[16][0] - ref_candidate[14][0]) ** 2 + (ref_candidate[16][1] - ref_candidate[14][1]) ** 2) ** 0.5
        l_head16_0 = ((candidate[16][0] - candidate[14][0]) ** 2 + (candidate[16][1] - candidate[14][1]) ** 2) ** 0.5

        head16_ratio = l_head16_ref / l_head16_0

        x_offset_head16 = (candidate[14][0]-candidate[16][0])*(1.-head16_ratio)
        y_offset_head16 = (candidate[14][1]-candidate[16][1])*(1.-head16_ratio)

        results_vis[0]['bodies']['candidate'][16,0] += x_offset_head16
        results_vis[0]['bodies']['candidate'][16,1] += y_offset_head16

        ########head17########
        l_head17_ref = ((ref_candidate[17][0] - ref_candidate[15][0]) ** 2 + (ref_candidate[17][1] - ref_candidate[15][1]) ** 2) ** 0.5
        l_head17_0 = ((candidate[17][0] - candidate[15][0]) ** 2 + (candidate[17][1] - candidate[15][1]) ** 2) ** 0.5

        head17_ratio = l_head17_ref / l_head17_0

        x_offset_head17 = (candidate[15][0]-candidate[17][0])*(1.-head17_ratio)
        y_offset_head17 = (candidate[15][1]-candidate[17][1])*(1.-head17_ratio)

        results_vis[0]['bodies']['candidate'][17,0] += x_offset_head17
        results_vis[0]['bodies']['candidate'][17,1] += y_offset_head17
        
        ########MovingAverage########
        
        ########left leg########
        l_ll1_ref = ((ref_candidate[8][0] - ref_candidate[9][0]) ** 2 + (ref_candidate[8][1] - ref_candidate[9][1]) ** 2) ** 0.5
        l_ll1_0 = ((candidate[8][0] - candidate[9][0]) ** 2 + (candidate[8][1] - candidate[9][1]) ** 2) ** 0.5
        ll1_ratio = l_ll1_ref / l_ll1_0

        x_offset_ll1 = (candidate[9][0]-candidate[8][0])*(ll1_ratio-1.)
        y_offset_ll1 = (candidate[9][1]-candidate[8][1])*(ll1_ratio-1.)

        results_vis[0]['bodies']['candidate'][9,0] += x_offset_ll1
        results_vis[0]['bodies']['candidate'][9,1] += y_offset_ll1
        results_vis[0]['bodies']['candidate'][10,0] += x_offset_ll1
        results_vis[0]['bodies']['candidate'][10,1] += y_offset_ll1
        results_vis[0]['bodies']['candidate'][19,0] += x_offset_ll1
        results_vis[0]['bodies']['candidate'][19,1] += y_offset_ll1

        l_ll2_ref = ((ref_candidate[9][0] - ref_candidate[10][0]) ** 2 + (ref_candidate[9][1] - ref_candidate[10][1]) ** 2) ** 0.5
        l_ll2_0 = ((candidate[9][0] - candidate[10][0]) ** 2 + (candidate[9][1] - candidate[10][1]) ** 2) ** 0.5
        ll2_ratio = l_ll2_ref / l_ll2_0

        x_offset_ll2 = (candidate[10][0]-candidate[9][0])*(ll2_ratio-1.)
        y_offset_ll2 = (candidate[10][1]-candidate[9][1])*(ll2_ratio-1.)

        results_vis[0]['bodies']['candidate'][10,0] += x_offset_ll2
        results_vis[0]['bodies']['candidate'][10,1] += y_offset_ll2
        results_vis[0]['bodies']['candidate'][19,0] += x_offset_ll2
        results_vis[0]['bodies']['candidate'][19,1] += y_offset_ll2

        ########right leg########
        l_rl1_ref = ((ref_candidate[11][0] - ref_candidate[12][0]) ** 2 + (ref_candidate[11][1] - ref_candidate[12][1]) ** 2) ** 0.5
        l_rl1_0 = ((candidate[11][0] - candidate[12][0]) ** 2 + (candidate[11][1] - candidate[12][1]) ** 2) ** 0.5
        rl1_ratio = l_rl1_ref / l_rl1_0

        x_offset_rl1 = (candidate[12][0]-candidate[11][0])*(rl1_ratio-1.)
        y_offset_rl1 = (candidate[12][1]-candidate[11][1])*(rl1_ratio-1.)

        results_vis[0]['bodies']['candidate'][12,0] += x_offset_rl1
        results_vis[0]['bodies']['candidate'][12,1] += y_offset_rl1
        results_vis[0]['bodies']['candidate'][13,0] += x_offset_rl1
        results_vis[0]['bodies']['candidate'][13,1] += y_offset_rl1
        results_vis[0]['bodies']['candidate'][18,0] += x_offset_rl1
        results_vis[0]['bodies']['candidate'][18,1] += y_offset_rl1

        l_rl2_ref = ((ref_candidate[12][0] - ref_candidate[13][0]) ** 2 + (ref_candidate[12][1] - ref_candidate[13][1]) ** 2) ** 0.5
        l_rl2_0 = ((candidate[12][0] - candidate[13][0]) ** 2 + (candidate[12][1] - candidate[13][1]) ** 2) ** 0.5
        rl2_ratio = l_rl2_ref / l_rl2_0

        x_offset_rl2 = (candidate[13][0]-candidate[12][0])*(rl2_ratio-1.)
        y_offset_rl2 = (candidate[13][1]-candidate[12][1])*(rl2_ratio-1.)

        results_vis[0]['bodies']['candidate'][13,0] += x_offset_rl2
        results_vis[0]['bodies']['candidate'][13,1] += y_offset_rl2
        results_vis[0]['bodies']['candidate'][18,0] += x_offset_rl2
        results_vis[0]['bodies']['candidate'][18,1] += y_offset_rl2

        offset = ref_candidate[1] - results_vis[0]['bodies']['candidate'][1]

        results_vis[0]['bodies']['candidate'] += offset[np.newaxis, :]
        results_vis[0]['faces'] += offset[np.newaxis, np.newaxis, :]
        results_vis[0]['hands'] += offset[np.newaxis, np.newaxis, :]

        for i in range(1, len(results_vis)):
            results_vis[i]['bodies']['candidate'][:,0] *= x_ratio
            results_vis[i]['bodies']['candidate'][:,1] *= y_ratio
            results_vis[i]['faces'][:,:,0] *= x_ratio
            results_vis[i]['faces'][:,:,1] *= y_ratio
            results_vis[i]['hands'][:,:,0] *= x_ratio
            results_vis[i]['hands'][:,:,1] *= y_ratio

            ########neck########
            x_offset_neck = (results_vis[i]['bodies']['candidate'][1][0]-results_vis[i]['bodies']['candidate'][0][0])*(1.-neck_ratio)
            y_offset_neck = (results_vis[i]['bodies']['candidate'][1][1]-results_vis[i]['bodies']['candidate'][0][1])*(1.-neck_ratio)

            results_vis[i]['bodies']['candidate'][0,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][0,1] += y_offset_neck
            results_vis[i]['bodies']['candidate'][14,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][14,1] += y_offset_neck
            results_vis[i]['bodies']['candidate'][15,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][15,1] += y_offset_neck
            results_vis[i]['bodies']['candidate'][16,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][16,1] += y_offset_neck
            results_vis[i]['bodies']['candidate'][17,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][17,1] += y_offset_neck

            ########shoulder2########
            

            x_offset_shoulder2 = (results_vis[i]['bodies']['candidate'][1][0]-results_vis[i]['bodies']['candidate'][2][0])*(1.-shoulder2_ratio)
            y_offset_shoulder2 = (results_vis[i]['bodies']['candidate'][1][1]-results_vis[i]['bodies']['candidate'][2][1])*(1.-shoulder2_ratio)

            results_vis[i]['bodies']['candidate'][2,0] += x_offset_shoulder2
            results_vis[i]['bodies']['candidate'][2,1] += y_offset_shoulder2
            results_vis[i]['bodies']['candidate'][3,0] += x_offset_shoulder2
            results_vis[i]['bodies']['candidate'][3,1] += y_offset_shoulder2
            results_vis[i]['bodies']['candidate'][4,0] += x_offset_shoulder2
            results_vis[i]['bodies']['candidate'][4,1] += y_offset_shoulder2
            results_vis[i]['hands'][1,:,0] += x_offset_shoulder2
            results_vis[i]['hands'][1,:,1] += y_offset_shoulder2

            ########shoulder5########

            x_offset_shoulder5 = (results_vis[i]['bodies']['candidate'][1][0]-results_vis[i]['bodies']['candidate'][5][0])*(1.-shoulder5_ratio)
            y_offset_shoulder5 = (results_vis[i]['bodies']['candidate'][1][1]-results_vis[i]['bodies']['candidate'][5][1])*(1.-shoulder5_ratio)

            results_vis[i]['bodies']['candidate'][5,0] += x_offset_shoulder5
            results_vis[i]['bodies']['candidate'][5,1] += y_offset_shoulder5
            results_vis[i]['bodies']['candidate'][6,0] += x_offset_shoulder5
            results_vis[i]['bodies']['candidate'][6,1] += y_offset_shoulder5
            results_vis[i]['bodies']['candidate'][7,0] += x_offset_shoulder5
            results_vis[i]['bodies']['candidate'][7,1] += y_offset_shoulder5
            results_vis[i]['hands'][0,:,0] += x_offset_shoulder5
            results_vis[i]['hands'][0,:,1] += y_offset_shoulder5

            ########arm3########

            x_offset_arm3 = (results_vis[i]['bodies']['candidate'][2][0]-results_vis[i]['bodies']['candidate'][3][0])*(1.-arm3_ratio)
            y_offset_arm3 = (results_vis[i]['bodies']['candidate'][2][1]-results_vis[i]['bodies']['candidate'][3][1])*(1.-arm3_ratio)

            results_vis[i]['bodies']['candidate'][3,0] += x_offset_arm3
            results_vis[i]['bodies']['candidate'][3,1] += y_offset_arm3
            results_vis[i]['bodies']['candidate'][4,0] += x_offset_arm3
            results_vis[i]['bodies']['candidate'][4,1] += y_offset_arm3
            results_vis[i]['hands'][1,:,0] += x_offset_arm3
            results_vis[i]['hands'][1,:,1] += y_offset_arm3

            ########arm4########

            x_offset_arm4 = (results_vis[i]['bodies']['candidate'][3][0]-results_vis[i]['bodies']['candidate'][4][0])*(1.-arm4_ratio)
            y_offset_arm4 = (results_vis[i]['bodies']['candidate'][3][1]-results_vis[i]['bodies']['candidate'][4][1])*(1.-arm4_ratio)

            results_vis[i]['bodies']['candidate'][4,0] += x_offset_arm4
            results_vis[i]['bodies']['candidate'][4,1] += y_offset_arm4
            results_vis[i]['hands'][1,:,0] += x_offset_arm4
            results_vis[i]['hands'][1,:,1] += y_offset_arm4

            ########arm6########

            x_offset_arm6 = (results_vis[i]['bodies']['candidate'][5][0]-results_vis[i]['bodies']['candidate'][6][0])*(1.-arm6_ratio)
            y_offset_arm6 = (results_vis[i]['bodies']['candidate'][5][1]-results_vis[i]['bodies']['candidate'][6][1])*(1.-arm6_ratio)

            results_vis[i]['bodies']['candidate'][6,0] += x_offset_arm6
            results_vis[i]['bodies']['candidate'][6,1] += y_offset_arm6
            results_vis[i]['bodies']['candidate'][7,0] += x_offset_arm6
            results_vis[i]['bodies']['candidate'][7,1] += y_offset_arm6
            results_vis[i]['hands'][0,:,0] += x_offset_arm6
            results_vis[i]['hands'][0,:,1] += y_offset_arm6

            ########arm7########

            x_offset_arm7 = (results_vis[i]['bodies']['candidate'][6][0]-results_vis[i]['bodies']['candidate'][7][0])*(1.-arm7_ratio)
            y_offset_arm7 = (results_vis[i]['bodies']['candidate'][6][1]-results_vis[i]['bodies']['candidate'][7][1])*(1.-arm7_ratio)

            results_vis[i]['bodies']['candidate'][7,0] += x_offset_arm7
            results_vis[i]['bodies']['candidate'][7,1] += y_offset_arm7
            results_vis[i]['hands'][0,:,0] += x_offset_arm7
            results_vis[i]['hands'][0,:,1] += y_offset_arm7

            ########head14########

            x_offset_head14 = (results_vis[i]['bodies']['candidate'][0][0]-results_vis[i]['bodies']['candidate'][14][0])*(1.-head14_ratio)
            y_offset_head14 = (results_vis[i]['bodies']['candidate'][0][1]-results_vis[i]['bodies']['candidate'][14][1])*(1.-head14_ratio)

            results_vis[i]['bodies']['candidate'][14,0] += x_offset_head14
            results_vis[i]['bodies']['candidate'][14,1] += y_offset_head14
            results_vis[i]['bodies']['candidate'][16,0] += x_offset_head14
            results_vis[i]['bodies']['candidate'][16,1] += y_offset_head14

            ########head15########

            x_offset_head15 = (results_vis[i]['bodies']['candidate'][0][0]-results_vis[i]['bodies']['candidate'][15][0])*(1.-head15_ratio)
            y_offset_head15 = (results_vis[i]['bodies']['candidate'][0][1]-results_vis[i]['bodies']['candidate'][15][1])*(1.-head15_ratio)

            results_vis[i]['bodies']['candidate'][15,0] += x_offset_head15
            results_vis[i]['bodies']['candidate'][15,1] += y_offset_head15
            results_vis[i]['bodies']['candidate'][17,0] += x_offset_head15
            results_vis[i]['bodies']['candidate'][17,1] += y_offset_head15

            ########head16########

            x_offset_head16 = (results_vis[i]['bodies']['candidate'][14][0]-results_vis[i]['bodies']['candidate'][16][0])*(1.-head16_ratio)
            y_offset_head16 = (results_vis[i]['bodies']['candidate'][14][1]-results_vis[i]['bodies']['candidate'][16][1])*(1.-head16_ratio)

            results_vis[i]['bodies']['candidate'][16,0] += x_offset_head16
            results_vis[i]['bodies']['candidate'][16,1] += y_offset_head16

            ########head17########
            x_offset_head17 = (results_vis[i]['bodies']['candidate'][15][0]-results_vis[i]['bodies']['candidate'][17][0])*(1.-head17_ratio)
            y_offset_head17 = (results_vis[i]['bodies']['candidate'][15][1]-results_vis[i]['bodies']['candidate'][17][1])*(1.-head17_ratio)

            results_vis[i]['bodies']['candidate'][17,0] += x_offset_head17
            results_vis[i]['bodies']['candidate'][17,1] += y_offset_head17

            # ########MovingAverage########

            ########left leg########
            x_offset_ll1 = (results_vis[i]['bodies']['candidate'][9][0]-results_vis[i]['bodies']['candidate'][8][0])*(ll1_ratio-1.)
            y_offset_ll1 = (results_vis[i]['bodies']['candidate'][9][1]-results_vis[i]['bodies']['candidate'][8][1])*(ll1_ratio-1.)

            results_vis[i]['bodies']['candidate'][9,0] += x_offset_ll1
            results_vis[i]['bodies']['candidate'][9,1] += y_offset_ll1
            results_vis[i]['bodies']['candidate'][10,0] += x_offset_ll1
            results_vis[i]['bodies']['candidate'][10,1] += y_offset_ll1
            results_vis[i]['bodies']['candidate'][19,0] += x_offset_ll1
            results_vis[i]['bodies']['candidate'][19,1] += y_offset_ll1



            x_offset_ll2 = (results_vis[i]['bodies']['candidate'][10][0]-results_vis[i]['bodies']['candidate'][9][0])*(ll2_ratio-1.)
            y_offset_ll2 = (results_vis[i]['bodies']['candidate'][10][1]-results_vis[i]['bodies']['candidate'][9][1])*(ll2_ratio-1.)

            results_vis[i]['bodies']['candidate'][10,0] += x_offset_ll2
            results_vis[i]['bodies']['candidate'][10,1] += y_offset_ll2
            results_vis[i]['bodies']['candidate'][19,0] += x_offset_ll2
            results_vis[i]['bodies']['candidate'][19,1] += y_offset_ll2

            ########right leg########

            x_offset_rl1 = (results_vis[i]['bodies']['candidate'][12][0]-results_vis[i]['bodies']['candidate'][11][0])*(rl1_ratio-1.)
            y_offset_rl1 = (results_vis[i]['bodies']['candidate'][12][1]-results_vis[i]['bodies']['candidate'][11][1])*(rl1_ratio-1.)

            results_vis[i]['bodies']['candidate'][12,0] += x_offset_rl1
            results_vis[i]['bodies']['candidate'][12,1] += y_offset_rl1
            results_vis[i]['bodies']['candidate'][13,0] += x_offset_rl1
            results_vis[i]['bodies']['candidate'][13,1] += y_offset_rl1
            results_vis[i]['bodies']['candidate'][18,0] += x_offset_rl1
            results_vis[i]['bodies']['candidate'][18,1] += y_offset_rl1


            x_offset_rl2 = (results_vis[i]['bodies']['candidate'][13][0]-results_vis[i]['bodies']['candidate'][12][0])*(rl2_ratio-1.)
            y_offset_rl2 = (results_vis[i]['bodies']['candidate'][13][1]-results_vis[i]['bodies']['candidate'][12][1])*(rl2_ratio-1.)

            results_vis[i]['bodies']['candidate'][13,0] += x_offset_rl2
            results_vis[i]['bodies']['candidate'][13,1] += y_offset_rl2
            results_vis[i]['bodies']['candidate'][18,0] += x_offset_rl2
            results_vis[i]['bodies']['candidate'][18,1] += y_offset_rl2

            results_vis[i]['bodies']['candidate'] += offset[np.newaxis, :]
            results_vis[i]['faces'] += offset[np.newaxis, np.newaxis, :]
            results_vis[i]['hands'] += offset[np.newaxis, np.newaxis, :]
    
    dwpose_woface_list = []
    for i in range(len(results_vis)):
        print("11111111111111", results_vis_copy[i]['bodies']['candidate'])
        print("22222222222222", results_vis[i]['bodies']['candidate'])
        orig  = results_vis_copy[i]['bodies']['candidate']     # shape (N_joints, 2)
        moved = results_vis[i]['bodies']['candidate']  # same shape
        # 1) compute per‐joint displacements
        displacements = np.linalg.norm(moved - orig, axis=1)  # length N_joints
        print("displacements", displacements)
        
        # 2) pick a metric: here the maximum movement
        max_disp = displacements.max()
        print("max_disp", max_disp)
        ratio    = max_disp / img_diag
        print("ratio", ratio)
        # 4) if too large, revert
        if ratio > movement_thresh:
            results_vis_used = results_vis_copy[i]
        else:
            results_vis_used = results_vis[i]
        try:
            dwpose_woface, dwpose_wface = module.draw_pose(results_vis_used, H=height, W=width, stick_width=stick_width,
                                                        draw_body=draw_body, draw_hands=draw_hands, hand_keypoint_size=hand_keypoint_size,
                                                        draw_feet=draw_feet, body_keypoint_size=body_keypoint_size, draw_head=draw_head)
            result = torch.from_numpy(dwpose_woface)
        except:
           result = torch.zeros((height, width, 3), dtype=torch.uint8)
        dwpose_woface_list.append(result)
    dwpose_woface_tensor = torch.stack(dwpose_woface_list, dim=0)

    dwpose_woface_ref_tensor = None
    if ref_image is not None:
        dwpose_woface_ref, dwpose_wface_ref = module.draw_pose(pose_ref, H=height, W=width, stick_width=stick_width,
                                                        draw_body=draw_body, draw_hands=draw_hands, hand_keypoint_size=hand_keypoint_size,
                                                        draw_feet=draw_feet, body_keypoint_size=body_keypoint_size, draw_head=draw_head)
        dwpose_woface_ref_tensor = torch.from_numpy(dwpose_woface_ref)

    return dwpose_woface_tensor, dwpose_woface_ref_tensor, results_vis_copy, pose_ref

def pose_extract(
    pose_images, ref_image, dwpose_model,
    height, width, score_threshold, stick_width,
    draw_body=True, draw_hands=True, hand_keypoint_size=4,
    draw_feet=True, body_keypoint_size=4,
    handle_not_detected="repeat", draw_head=True
):
    """
    Hierarchical bone‐by‐bone retargeting with invalid‐index filtering.
    """
    # 1) detect reference pose once
    pose_ref = None
    if ref_image is not None:
        try:
            pose_ref = dwpose_model(ref_image.squeeze(0), score_threshold=score_threshold)
        except:
            raise ValueError("No pose detected in reference image")
        ref_cand = pose_ref['bodies']['candidate']
        # Precompute reference torso length (neck to hips midpoint)
        neck_ref = ref_cand[1]
        hip_mid_ref = 0.5 * (ref_cand[8] + ref_cand[11])
        torso_len_ref = np.linalg.norm(hip_mid_ref - neck_ref)
        # Precompute reference head length (nose to neck)
        head_len_ref = np.linalg.norm(ref_cand[0] - neck_ref)
        head_to_torso_ref = head_len_ref / torso_len_ref

    # 2) detect all input frames (with “repeat” fallback)
    raw_vis = []
    prev = None
    for img in pose_images:
        try:
            p = dwpose_model(img, score_threshold=score_threshold)
            prev = p
        except:
            if prev is None:
                # empty template if nothing ever detected
                zero_body = np.zeros((20,2), dtype=float)
                zero_face = np.zeros((1,68,2), dtype=float)
                zero_hand = np.zeros((2,21,2), dtype=float)
                p = {'bodies': {'candidate': zero_body},
                     'faces': zero_face,
                     'hands': zero_hand}
            else:
                p = prev
        raw_vis.append(p)

    # if no reference, skip retargeting
    if pose_ref is None:
        results_vis = raw_vis
    else:
        # 3) build the edge list
        limbSeq = [
            [2, 3],   # Neck to Right Shoulder
            [2, 6],  # Neck to Left Shoulder
            [3, 4], # Right Shoulder to Right Elbow
            [4, 5], # Right Elbow to Right Wrist
            [6, 7], # Left Shoulder to Left Elbow
            [7, 8],  # Left Elbow to Left Wrist
            [2, 9],   # Neck to Right Hip
            [9, 10], # Right Hip to Right Knee
            [10, 11], # Right Knee to Right Ankle
            [2, 12], # Neck to Left Hip
            [12, 13], # Left Hip to Left Knee
            [13, 14], # Left Knee to Left Ankle
            [14, 19], # Left Ankle to Right Foot
            [11, 20], # Right Ankle to Left Foot
        ]
        headSeq = [
            [2, 1], # Neck to Nose
            [1, 15], # Nose to Right Eye
            [15, 17], # Right Eye to Right Ear
            [1, 16], # Nose to Left Eye
            [16, 18], # Left Eye to Left Ear
            [3, 17], # Right Shoulder to Right Ear
            [6, 18],   # Left Shoulder to Left Ear
        ]
        # edges = limbSeq + headSeq
        # Don't stretch the head, only the body
        edges = limbSeq
        PARENT = { b-1: a-1 for a,b in edges }
        ROOT = 2-1  # neck

        # Build children map for hierarchical propagation
        children_map = {}
        for child, parent in PARENT.items():
            children_map.setdefault(parent, []).append(child)

        # symmetric pairs (0-based) for ratio averaging
        symmetric_pairs = [
            (3-1, 6-1),  # shoulders
            (4-1, 7-1),  # upper arms
            (5-1, 8-1),  # forearms
            (9-1,12-1),  # hips
            (10-1,13-1), # thighs
            (11-1,14-1)  # calves
        ]

        # Define wrist indices (0-based) for per-hand translation
        RIGHT_WRIST = 5 - 1
        LEFT_WRIST  = 8 - 1
        # Define ankle indices for foot baseline anchoring (0-based)
        RIGHT_ANKLE = 11 - 1
        LEFT_ANKLE  = 14 - 1
        # Head cluster indices (0-based): nose, eyes, ears
        HEAD_IDS = [id - 1 for id in [1, 15, 16, 17, 18]]

        # 4) Retarget each frame using its own ratios
        results_vis = []
        for frame in raw_vis:
            src_cand = frame['bodies']['candidate']
            neck_src = src_cand[ROOT]
            # compute source torso & head lengths
            hip_mid_src = 0.5 * (src_cand[8] + src_cand[11])
            torso_len_src = np.linalg.norm(hip_mid_src - neck_src)
            head_len_src = np.linalg.norm(src_cand[0] - neck_src)
            head_to_torso_src = head_len_src / torso_len_src
            # head block scale factor
            head_scale = head_to_torso_ref / head_to_torso_src

            # per-bone ratios
            ratio = {}
            for c,p in PARENT.items():
                v_src = src_cand[c] - src_cand[p]
                v_ref = ref_cand[c] - ref_cand[p]
                r = np.linalg.norm(v_ref) / (np.linalg.norm(v_src) + 1e-8)
                ratio[c] = r
            # enforce symmetry
            for a,b in symmetric_pairs:
                m = 0.5*(ratio[a]+ratio[b])
                ratio[a]=ratio[b]=m
            # clamp if desired
            for k in ratio: ratio[k] = np.clip(ratio[k], 0.5, 2.0)

            # retarget
            out = np.zeros_like(src_cand)
            out[ROOT] = src_cand[ROOT]
            def dfs(u):
                for w in children_map.get(u,[]):
                    out[w] = out[u] + ratio[w] * (src_cand[w] - src_cand[u])
                    dfs(w)
            dfs(ROOT)

            # foot baseline vertical adjust
            orig_feet = max(src_cand[RIGHT_ANKLE,1], src_cand[LEFT_ANKLE,1])
            new_feet  = max(out[RIGHT_ANKLE,1], out[LEFT_ANKLE,1])
            out[:,1] += (orig_feet - new_feet)

            neck_off = out[ROOT] - neck_src
            for hid in HEAD_IDS:
                # restore head block
                out[hid] = src_cand[hid]
                out[hid] += neck_off

            # assemble frame
            new_frame = deepcopy(frame)
            new_frame['bodies']['candidate'] = out
            # translate faces by neck offset
            new_frame['faces'] += neck_off[np.newaxis, np.newaxis, :]
            # translate hands
            rw_off = out[RIGHT_WRIST] - src_cand[RIGHT_WRIST]
            lw_off = out[LEFT_WRIST]  - src_cand[LEFT_WRIST]
            new_frame['hands'][0] += lw_off
            new_frame['hands'][1] += rw_off

            results_vis.append(new_frame)

    # 5) Draw all retargeted poses
    dwpose_list = []
    for f in tqdm(results_vis, desc="Redraw retargeted poses"):
        wo, _ = module.draw_pose(
            f, H=height, W=width, stick_width=stick_width,
            draw_body=draw_body, draw_hands=draw_hands,
            hand_keypoint_size=hand_keypoint_size,
            draw_feet=draw_feet, body_keypoint_size=body_keypoint_size,
            draw_head=draw_head
        )
        dwpose_list.append(torch.from_numpy(wo))
    batch = torch.stack(dwpose_list, dim=0)

    # 6) Optionally draw the reference pose
    ref_tensor = None
    if pose_ref is not None:
        wo_ref, _ = module.draw_pose(
            pose_ref, H=height, W=width, stick_width=stick_width,
            draw_body=draw_body, draw_hands=draw_hands,
            hand_keypoint_size=hand_keypoint_size,
            draw_feet=draw_feet, body_keypoint_size=body_keypoint_size,
            draw_head=draw_head
        )
        ref_tensor = torch.from_numpy(wo_ref)

    return batch, ref_tensor

def pose_to_serializable(pose_dict):
    """
    Recursively turn a dwpose_model output dict
    (with np.ndarrays and/or torch.Tensors) into pure
    Python structures of lists, so it can be JSON-dumped.
    """
    serial = {}
    for k, v in pose_dict.items():
        if isinstance(v, dict):
            serial[k] = pose_to_serializable(v)
        elif isinstance(v, np.ndarray):
            serial[k] = v.tolist()
        elif isinstance(v, torch.Tensor):
            serial[k] = v.cpu().numpy().tolist()
        elif isinstance(v, list):
            # if you've ever nested lists of arrays
            serial[k] = [ pose_to_serializable(x) if isinstance(x, dict)
                          else (x.tolist() if isinstance(x, np.ndarray)
                                else x)
                          for x in v ]
        else:
            serial[k] = v  # e.g. ints, floats
    return serial

def save_all_poses_as_json(raw_vis, pose_ref, out_path_inputs, out_path_ref=None):
    """
    raw_vis      : list of dicts returned by dwpose_model for each input image
    pose_ref     : single dict for your reference image (or None)
    out_path_inputs: filename to write input poses JSON
    out_path_ref   : filename to write reference pose JSON (optional)
    """
    # Convert and dump input frames
    serial_inputs = [ pose_to_serializable(frame) for frame in raw_vis ]
    with open(out_path_inputs, 'w') as f:
        json.dump(serial_inputs, f, indent=2)
    
    # If you want the ref pose too:
    if pose_ref is not None and out_path_ref is not None:
        with open(out_path_ref, 'w') as f:
            json.dump(pose_to_serializable(pose_ref), f, indent=2)

def serializable_to_pose(obj, to_torch: bool = False):
    """
    Recursively convert a JSON‐loaded structure of dicts/lists
    back into dicts with np.ndarrays (or torch.Tensors).
    """
    if isinstance(obj, dict):
        return {k: serializable_to_pose(v, to_torch) for k, v in obj.items()}

    # If it’s a list of numbers or list of lists, turn it into an array
    if isinstance(obj, list):
        # detect if this list is numeric (homogeneous) and should be an array
        if all(isinstance(x, (int, float)) for x in obj):
            arr = np.array(obj, dtype=float)
            return torch.from_numpy(arr) if to_torch else arr

        if all(isinstance(x, list) for x in obj):
            # nested lists → multi-D array
            arr = np.array(obj, dtype=float)
            return torch.from_numpy(arr) if to_torch else arr

        # otherwise it’s a heterogenous/list-of-poses, recurse elementwise
        return [serializable_to_pose(x, to_torch) for x in obj]

    # base case: leave ints, floats, strings alone
    return obj

def load_poses_from_json(path_inputs: str,
                         path_ref: str = None,
                         to_torch: bool = False):
    """
    Load your saved input‐poses JSON and (optionally) reference‐pose JSON,
    and reconstruct the same dicts you got from dwpose_model.
    
    Returns:
      raw_vis, pose_ref
      – raw_vis: List[dict]  (one dict per frame)
      – pose_ref: dict or None
    """
    # 1) load inputs
    with open(path_inputs, 'r') as f:
        data = json.load(f)
    raw_vis = [serializable_to_pose(frame, to_torch) for frame in data]

    # 2) load reference if requested
    pose_ref = None
    if path_ref:
        with open(path_ref, 'r') as f:
            ref_data = json.load(f)
        pose_ref = serializable_to_pose(ref_data, to_torch)

    return raw_vis, pose_ref


class MyWanVideoUniAnimateDWPoseDetector:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "pose_images": ("IMAGE", {"tooltip": "Pose images"}),
                "score_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Score threshold for pose detection"}),
                "stick_width": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1, "tooltip": "Stick width for drawing keypoints"}),
                "draw_body": ("BOOLEAN", {"default": True, "tooltip": "Draw body keypoints"}),
                "body_keypoint_size": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1, "tooltip": "Body keypoint size"}),
                "draw_feet": ("BOOLEAN", {"default": True, "tooltip": "Draw feet keypoints"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Draw hand keypoints"}),
                "hand_keypoint_size": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1, "tooltip": "Hand keypoint size"}),
                "colorspace": (["RGB", "BGR"], {"tooltip": "Color space for the output image"}),
                "handle_not_detected": (["empty", "repeat"], {"default": "empty", "tooltip": "How to handle undetected poses, empty inserts black and repeat inserts previous detection"}),
                "draw_head": ("BOOLEAN", {"default": True, "tooltip": "Draw head keypoints"}),
            },
            "optional": {
                "reference_pose_image": ("IMAGE", {"tooltip": "Reference pose image"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", )
    RETURN_NAMES = ("poses", "reference_pose",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, pose_images, score_threshold, stick_width, reference_pose_image=None, draw_body=True, body_keypoint_size=4, 
                draw_feet=True, draw_hands=True, hand_keypoint_size=4, colorspace="RGB", handle_not_detected="empty", draw_head=True):

        device = mm.get_torch_device()
        
        #model loading
        dw_pose_model = "dw-ll_ucoco_384_bs5.torchscript.pt"
        yolo_model = "yolox_l.torchscript.pt"

        script_directory = os.path.dirname(os.path.abspath(__file__))
        model_base_path = os.path.join(script_directory, "models", "DWPose")

        model_det=os.path.join(model_base_path, yolo_model)
        model_pose=os.path.join(model_base_path, dw_pose_model)

        if not os.path.exists(model_det):
            # log.info(f"Downloading yolo model to: {model_base_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="hr16/yolox-onnx", 
                                allow_patterns=[f"*{yolo_model}*"],
                                local_dir=model_base_path, 
                                local_dir_use_symlinks=False)
            
        if not os.path.exists(model_pose):
            # log.info(f"Downloading dwpose model to: {model_base_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="hr16/DWPose-TorchScript-BatchSize5", 
                                allow_patterns=[f"*{dw_pose_model}*"],
                                local_dir=model_base_path, 
                                local_dir_use_symlinks=False)

        if not hasattr(self, "det") or not hasattr(self, "pose"):
            self.det = torch.jit.load(model_det, map_location=device)
            self.pose = torch.jit.load(model_pose, map_location=device)
            self.dwpose_detector = module.DWposeDetector(self.det, self.pose) 

        #model inference
        height, width = pose_images.shape[1:3]
        
        pose_np = pose_images.cpu().numpy() * 255
        ref_np = None
        if reference_pose_image is not None:
            ref = reference_pose_image
            ref_np = ref.cpu().numpy() * 255

        poses, reference_pose = pose_extract(pose_np, ref_np, self.dwpose_detector, height, width, score_threshold, stick_width=stick_width,
                                             draw_body=draw_body, body_keypoint_size=body_keypoint_size, draw_feet=draw_feet, 
                                             draw_hands=draw_hands, hand_keypoint_size=hand_keypoint_size, handle_not_detected=handle_not_detected, draw_head=draw_head)
        poses = poses / 255.0
        if reference_pose_image is not None:
            reference_pose = reference_pose.unsqueeze(0) / 255.0
        else:
            reference_pose = torch.zeros(1, 64, 64, 3, device=torch.device("cpu"))

        if colorspace == "BGR":
            poses=torch.flip(poses, dims=[-1])

        # with open("/workspace/poses.json", "w") as f:
        #     json.dump(results_vis_copy, f, indent=4)
        # with open("/workspace/pose_ref.json", "w") as f:
        #     json.dump(pose_ref, f, indent=4)
        # save_all_poses_as_json(results_vis_copy, pose_ref, "/workspace/poses.json", "/workspace/pose_ref.json")

        return (poses, reference_pose, )