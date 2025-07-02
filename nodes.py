import numpy as np
import torch
import math
from PIL import Image, ImageDraw
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

class ScaleSkeletonsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeletons": ("IMAGE", {"multiple": True}),
                "mask":      ("MASK",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "scale_skeletons"
    CATEGORY = "Custom/Transform"

    def scale_skeletons(self, skeletons, mask):
        # --- 1) Mask → H×W bool array + bbox/center ---
        mask_np = mask.detach().cpu().numpy() if torch.is_tensor(mask) else mask
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_arr = mask_np[0]
        else:
            mask_arr = mask_np
        mask_bin = mask_arr > 0.5
        ys, xs = np.where(mask_bin)
        if ys.size == 0:
            print("⚠️ Mask empty — returning original skeleton list")
            return (skeletons,)

        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        ref_w, ref_h = x1 - x0, y1 - y0
        mask_cx, mask_cy = x0 + ref_w/2, y0 + ref_h/2
        # print(f"Mask bbox: ({x0},{y0})→({x1},{y1}), size={ref_w}×{ref_h}, center=({mask_cx:.1f},{mask_cy:.1f})")

        # --- 2) Compute scale from first skeleton ---
        sk0 = skeletons[0]
        sk0_np = sk0.detach().cpu().numpy() if torch.is_tensor(sk0) else sk0
        # channel‐first → channel‐last if needed
        if sk0_np.ndim == 3 and sk0_np.shape[0] in (1,3,4):
            sk0_arr = np.transpose(sk0_np, (1,2,0))
        else:
            sk0_arr = sk0_np

        # convert float [0..1] → uint8 [0..255]
        if sk0_arr.dtype != np.uint8:
            sk0_uint8 = (np.clip(sk0_arr, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            sk0_uint8 = sk0_arr

        # find its nonzero‐alpha bbox
        alpha0 = np.any(sk0_uint8[...,:3] != 0, axis=2)
        ys2, xs2 = np.where(alpha0)
        sy0, sy1 = ys2.min(), ys2.max()
        sx0, sx1 = xs2.min(), xs2.max()
        sk_w, sk_h = sx1 - sx0, sy1 - sy0
        scale = min(ref_w/sk_w, ref_h/sk_h)
        print(f"First‐frame skeleton bbox: ({sx0},{sy0})→({sx1},{sy1}), size={sk_w}×{sk_h}, scale={scale:.3f}")

        H, W = sk0_uint8.shape[:2]
        outputs = []

        # --- 3) Loop every skeleton frame ---
        for idx, sk in enumerate(skeletons):
            sk_np = sk.detach().cpu().numpy() if torch.is_tensor(sk) else sk
            if sk_np.ndim == 3 and sk_np.shape[0] in (1,3,4):
                sk_arr = np.transpose(sk_np, (1,2,0))
            else:
                sk_arr = sk_np

            # to uint8
            if sk_arr.dtype != np.uint8:
                sk_uint8 = (np.clip(sk_arr, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                sk_uint8 = sk_arr

            # build RGBA
            if sk_uint8.ndim == 2:
                # gray→RGB + alpha
                alpha = (sk_uint8 > 0).astype(np.uint8) * 255
                rgba = np.dstack([sk_uint8]*3 + [alpha])
            elif sk_uint8.shape[2] == 3:
                alpha = (np.any(sk_uint8[...,:3] != 0, axis=2).astype(np.uint8) * 255)
                rgba = np.dstack([sk_uint8, alpha])
            else:
                rgba = sk_uint8  # already RGBA

            # crop → resize → paste
            region = rgba[sy0:sy1, sx0:sx1]
            # print(f"Frame {idx}: cropped region {region.shape}")
            pil = Image.fromarray(region, "RGBA")
            # uniform-scale for human
            # new_w, new_h = int(sk_w * scale), int(sk_h * scale)
            # resized = pil.resize((new_w, new_h), Image.BILINEAR)
            # paste_x = int(mask_cx - new_w / 2)
            # paste_y = int(mask_cy - new_h / 2)

            # non-uniform scale for animals
            # new non‐uniform code: stretch to exactly the mask’s box
            new_w, new_h = ref_w, ref_h
            resized = pil.resize((new_w, new_h), Image.BILINEAR)
            paste_x, paste_y = x0, y0

            # print(f"Frame {idx}: resizing → {new_w}×{new_h}, paste at ({paste_x},{paste_y})")

            canvas = Image.new("RGBA", (W, H), (0,0,0,255))
            canvas.paste(resized, (paste_x, paste_y), resized)

            out_np = np.array(canvas)  # uint8 H×W×4
            nonzero = np.count_nonzero(out_np[...,3])
            # print(f"Frame {idx}: nonzero alpha pixels after paste = {nonzero}")

            # channel-last float [0..1]
            out_t = torch.from_numpy(out_np.astype(np.float32) / 255.0)
            outputs.append(out_t)

        # print(f"🔍 Total frames output: {len(outputs)}")
        # instead of returning the raw list, stack into one tensor:
        video_tensor = torch.stack(outputs, dim=0)  # shape: (num_frames, H, W, 4)
        return (video_tensor,)

# Mapping of body parts to keypoint indices
BODY_PART_INDEXES = {
    "Head":     (16, 14, 0, 15, 17),
    "Neck":     (0, 1),
    "Shoulder": (2, 5),
    "Torso":    (2, 5, 8, 11),
    "RArm":     (2, 3),
    "RForearm": (3, 4),
    "LArm":     (5, 6),
    "LForearm": (6, 7),
    "RThigh":   (8, 9),
    "RLeg":     (9, 10),
    "LThigh":   (11, 12),
    "LLeg":     (12, 13),
}

class CuteSkeletonNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "poses":           ("POSE_KEYPOINT", {"multiple": True}),
                "head_scale":      ("FLOAT", {"default": 1.3,   "min": 0.01, "max": 5.0, "step": 0.01}),
                "neck_scale":      ("FLOAT", {"default": 1.0,   "min": 0.01, "max": 5.0, "step": 0.01}),
                "shoulder_scale": ("FLOAT", {"default": 1.0,   "min": 0.01, "max": 5.0, "step": 0.01}),
                "torso_scale":     ("FLOAT", {"default": 1.0,   "min": 0.01, "max": 5.0, "step": 0.01}),
                "rarm_scale":      ("FLOAT", {"default": 0.8,   "min": 0.01, "max": 5.0, "step": 0.01}),
                "rforearm_scale":  ("FLOAT", {"default": 0.8,   "min": 0.01, "max": 5.0, "step": 0.01}),
                "larm_scale":      ("FLOAT", {"default": 0.8,   "min": 0.01, "max": 5.0, "step": 0.01}),
                "lforearm_scale":  ("FLOAT", {"default": 0.8,   "min": 0.01, "max": 5.0, "step": 0.01}),
                "rthigh_scale":    ("FLOAT", {"default": 0.7,   "min": 0.01, "max": 5.0, "step": 0.01}),
                "rleg_scale":      ("FLOAT", {"default": 1.0,   "min": 0.01, "max": 5.0, "step": 0.01}),
                "lthigh_scale":    ("FLOAT", {"default": 0.7,   "min": 0.01, "max": 5.0, "step": 0.01}),
                "lleg_scale":      ("FLOAT", {"default": 1.0,   "min": 0.01, "max": 5.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("POSE_KEYPOINT",)
    FUNCTION = "make_cute_skeletons"
    CATEGORY = "Custom/Transform"

    def make_cute_skeletons(
        self,
        poses,
        head_scale, neck_scale, shoulder_scale, torso_scale,
        rarm_scale, rforearm_scale, larm_scale, lforearm_scale,
        rthigh_scale, rleg_scale, lthigh_scale, lleg_scale
    ):
        outputs = []
        # Map of scale values by region name
        scales = {
            "Head":     head_scale,
            "Neck":     neck_scale,
            "Shoulder": shoulder_scale,
            "Torso":    torso_scale,
            "RArm":     rarm_scale,
            "RForearm": rforearm_scale,
            "LArm":     larm_scale,
            "LForearm": lforearm_scale,
            "RThigh":   rthigh_scale,
            "RLeg":     rleg_scale,
            "LThigh":   lthigh_scale,
            "LLeg":     lleg_scale,
        }
        for pose in poses:
            new_pose = {"people": []}
            # Preserve canvas dims if present
            if "canvas_width" in pose:
                new_pose["canvas_width"] = pose["canvas_width"]
            if "canvas_height" in pose:
                new_pose["canvas_height"] = pose["canvas_height"]

            # Process each person in the frame
            for person in pose.get("people", []):
                flat = person.get("pose_keypoints_2d", [])
                pts = np.array(flat, dtype=float).reshape(-1, 3)
                warped = pts.copy()

                # Apply scaling per body part
                for region, idxs in BODY_PART_INDEXES.items():
                    scale = scales.get(region, 1.0)
                    if scale != 1.0:
                        coords = warped[list(idxs), :2]
                        center = coords.mean(axis=0)
                        warped[list(idxs), :2] = center + (coords - center) * scale

                # Flatten back
                new_flat = warped.flatten().tolist()
                new_person = {"pose_keypoints_2d": new_flat}
                new_pose["people"].append(new_person)

            outputs.append(new_pose)
        return (outputs,)

class CatPoseRetargetNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "human_poses": ("POSE_KEYPOINT", {"multiple": True}),
                "cat_pose":    ("POSE_KEYPOINT",),
            }
        }
    RETURN_TYPES = ("POSE_KEYPOINT",)
    FUNCTION = "retarget_poses"
    CATEGORY = "Custom/Transform"

    @staticmethod
    def rotate2d(v, angle):
        """Rotate 2D vector v by angle radians"""
        c, s = math.cos(angle), math.sin(angle)
        return np.array([v[0]*c - v[1]*s, v[0]*s + v[1]*c])

    def retarget_poses(self, human_poses, cat_pose):
        # Unwrap cat pose
        cat_dict = cat_pose[0] if isinstance(cat_pose, list) else cat_pose
        cat_list = cat_dict.get("animals", [[]])[0]
        cat_kps = np.array(cat_list, dtype=float).reshape(-1, 3)
        # Modify input cat_pose: set 5th keypoint root of tail (index 4)
        # as midpoint of 12th (11) and 15th (14) left and right hips
        idx5 = 4  # 0-based
        print(cat_kps[idx5])
        idx12 = 11
        idx15 = 14
        midpoint = (cat_kps[idx12, :2] + cat_kps[idx15, :2]) / 2
        cat_kps[idx5, :2] = midpoint
       
        # Define human and cat index mappings
        human_parts = {
            'front_left':  (5, 6, 7),   # LShoulder, LElbow, LWrist
            'front_right': (2, 3, 4),   # RShoulder, RElbow, RWrist
            'hind_left':   (11,12,13),  # LHip, LKnee, LAnkle
            'hind_right':  (8, 9,10),   # RHip, RKnee, RAnkle
        }
        cat_parts = {
            'front_left':  (5, 6, 7),
            'front_right': (8, 9,10),
            'hind_left':   (11,12,13),
            'hind_right':  (14,15,16),
        }

        outputs = []
        for frame_idx, human in enumerate(human_poses):
            h_flat = human.get('people', [{}])[0].get('pose_keypoints_2d', [])
            human_kps = np.array(h_flat, dtype=float).reshape(-1,3)
            print(f"[DEBUG] Frame {frame_idx}")
            xs = human_kps[:,0]
            ys = human_kps[:,1]
            hxmin, hxmax = xs.min(), xs.max()
            hymin, hymax = ys.min(), ys.max()
            hwidth = hxmax - hxmin if hxmax > hxmin else 1.0
            hheight = hymax - hymin if hymax > hymin else 1.0
            # Compute cat bounding box
            cat_xs = cat_kps[:,0]
            cat_ys = cat_kps[:,1]
            cxmin, cxmax = cat_xs.min(), cat_xs.max()
            cymin, cymax = cat_ys.min(), cat_ys.max()
            cwidth = cxmax - cxmin if cxmax > cxmin else 1.0
            cheight = cymax - cymin if cymax > cymin else 1.0
            # Map human hip positions normalized into cat bbox space
            # human left hip idx11, cat idx11
            hL = human_kps[11,:2]
            nx = (hL[0] - hxmin) / hwidth
            ny = (hL[1] - hymin) / hheight
            cat_kps[11, :2] = np.array([cxmin + nx * cwidth, cymin + ny * cheight])
            # human right hip idx8, cat idx14
            hR = human_kps[8,:2]
            nx = (hR[0] - hxmin) / hwidth
            ny = (hR[1] - hymin) / hheight
            cat_kps[14, :2] = np.array([cxmin + nx * cwidth, cymin + ny * cheight])
            print(f"[DEBUG] Normalized human hips to cat bbox: left={cat_kps[11,:2]}, right={cat_kps[14,:2]}")
            # Create working copy for this frame
            new_cat = cat_kps.copy()

            # retarget each leg with analytic two-bone IK
            for leg, (h1,h2,h3) in human_parts.items():
                cp1,cp2,cp3 = cat_parts[leg]
                origin = cat_kps[cp1,:2]
                # human vectors
                v1 = human_kps[h2,:2] - human_kps[h1,:2]
                v2 = human_kps[h3,:2] - human_kps[h2,:2]
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1<1e-6 or n2<1e-6:
                    print(f"[DEBUG] {leg} skipped due to zero-length")
                    continue
                u1 = v1/n1
                u2 = v2/n2
                # compute bend angle
                dot = np.clip(np.dot(u1,u2), -1.0, 1.0)
                angle = math.acos(dot)
                cross = u1[0]*u2[1] - u1[1]*u2[0]
                sign = 1.0 if cross>=0 else -1.0
                # cat bone lengths
                L1 = np.linalg.norm(cat_kps[cp2,:2] - origin)
                L2 = np.linalg.norm(cat_kps[cp3,:2] - cat_kps[cp2,:2])
                # new positions
                elbow = origin + u1 * L1
                paw   = elbow + self.rotate2d(u1, sign*angle) * L2
                print(f"[DEBUG] {leg} elbow={elbow}, paw={paw}")
                new_cat[cp2,:2] = elbow
                new_cat[cp3,:2] = paw

            # retarget head
            h_nose = human_kps[0,:2]
            h_neck = human_kps[1,:2]
            v_head = h_nose - h_neck
            n_head = np.linalg.norm(v_head)
            if n_head>1e-6:
                dir_h = v_head/n_head
                idx_nose = 2  # 0-based
                idx_neck = 3
                Lh = np.linalg.norm(cat_kps[idx_nose,:2]-cat_kps[idx_neck,:2])
                new_cat[idx_nose,:2] = cat_kps[idx_neck,:2] + dir_h*Lh
                print(f"[DEBUG] head moved to {new_cat[idx_nose,:2]}")

            # retarget torso
            h_mid = (human_kps[11,:2]+human_kps[12,:2])/2
            v_torso = h_mid - h_neck
            n_t = np.linalg.norm(v_torso)
            if n_t>1e-6:
                dir_t = v_torso/n_t
                idx_tail = 4
                Lr = np.linalg.norm(cat_kps[idx_tail,:2]-cat_kps[idx_neck,:2])
                new_cat[idx_tail,:2] = cat_kps[idx_neck,:2] + dir_t*Lr
                print(f"[DEBUG] torso moved to {new_cat[idx_tail,:2]}")
            
            # Modify input cat_pose: set 5th keypoint root of tail (index 4)
            # as midpoint of 12th (11) and 15th (14) left and right hips
            idx5 = 4
            idx12 = 11
            idx15 = 14
            midpoint = (new_cat[idx12,:2] + new_cat[idx15,:2]) / 2
            new_cat[idx5,:2] = midpoint

            out = {'animals':[new_cat.tolist()]}
            for k in('canvas_width','canvas_height'):
                if k in cat_dict:
                    out[k]=cat_dict[k]
            outputs.append(out)

        return (outputs,)