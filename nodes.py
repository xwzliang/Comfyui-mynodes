import numpy as np
import torch
import math
from PIL import Image, ImageDraw
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

def to_pil(img):
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1,3,4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[...,0]
        if arr.dtype.kind == 'f':
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            mode = 'L'
        elif arr.ndim == 3 and arr.shape[2] == 3:
            mode = 'RGB'
        elif arr.ndim == 3 and arr.shape[2] == 4:
            mode = 'RGBA'
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")
        return Image.fromarray(arr, mode)
    elif isinstance(img, Image.Image):
        return img
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")


def to_tensor(arr):
    img_arr = arr.astype(np.float32) / 255.0
    if img_arr.ndim == 2:
        img_arr = img_arr[..., None]
    tensor = torch.from_numpy(img_arr)
    if tensor.ndim == 3:
        # HWC -> CHW
        tensor = tensor.permute(2, 0, 1)
    return tensor


class ScaleImagesToMaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":      ("IMAGE", {"multiple": True}),
                "mask":        ("MASK", {}),
                "scale_width": ("BOOLEAN", {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "scale_images"
    CATEGORY = "Custom/Transform"

    def scale_images(self, images, mask, scale_width, debug=False):
        # --- Compute global scale from first image and mask ---
        mask_pil = to_pil(mask).convert('L')
        mask_arr = np.array(mask_pil)
        ys_m, xs_m = np.nonzero(mask_arr > 0)
        if ys_m.size == 0:
            if debug: print("Mask empty, returning originals.")
            return (torch.stack([to_tensor(np.array(to_pil(img))) for img in images], dim=0),)
        y0_m, y1_m = ys_m.min(), ys_m.max()
        x0_m, x1_m = xs_m.min(), xs_m.max()
        ref_w, ref_h = x1_m - x0_m, y1_m - y0_m
        if debug: print(f"Mask bbox: ({x0_m},{y0_m})→({x1_m},{y1_m}), size={ref_w}×{ref_h}")

        # First image crop bbox
        pil0 = to_pil(images[0])
        arr0 = np.array(pil0)
        mask0 = np.any(arr0[..., :3] > 0, axis=-1)
        ys0, xs0 = np.nonzero(mask0)
        if ys0.size == 0:
            if debug: print("First image non-black empty, returning originals.")
            return (torch.stack([to_tensor(np.array(pil0)) for pil0 in images]),)
        sy0, sy1 = ys0.min(), ys0.max()
        sx0, sx1 = xs0.min(), xs0.max()
        sk_w, sk_h = sx1 - sx0, sy1 - sy0
        scale_y = ref_h / sk_h
        scale_x = ref_w / sk_w if scale_width else scale_y
        if debug: print(f"Global scales from first image: scale_x={scale_x:.3f}, scale_y={scale_y:.3f}")

        outputs = []
        # Process each image individually
        for idx, img in enumerate(images):
            pil = to_pil(img)
            arr = np.array(pil)
            mask_i = np.any(arr[..., :3] > 0, axis=-1)
            ys_i, xs_i = np.nonzero(mask_i)
            if ys_i.size == 0:
                if debug: print(f"Image {idx}: no non-black region, returning original.")
                outputs.append(np.array(pil).astype(np.float32)/255.0)
                continue
            sy0_i, sy1_i = ys_i.min(), ys_i.max()
            sx0_i, sx1_i = xs_i.min(), xs_i.max()
            region_w, region_h = sx1_i - sx0_i, sy1_i - sy0_i

            # New size
            new_w = int(region_w * scale_x)
            new_h = int(region_h * scale_y)
            # Center of original region
            cx = (sx0_i + sx1_i) / 2.0
            cy = (sy0_i + sy1_i) / 2.0
            # Compute top-left offset
            offset_x = int(cx - new_w/2)
            offset_y = int(y1_m - new_h)
            if debug: print(f"Image {idx}: region ({sx0_i},{sy0_i})→({sx1_i},{sy1_i}), new size={new_w}×{new_h}, offset=({offset_x},{offset_y})")

            # Crop, resize, paste
            region = pil.crop((sx0_i, sy0_i, sx1_i, sy1_i))
            region_scaled = region.resize((new_w, new_h), resample=Image.BICUBIC)
            # canvas = pil.copy()
            W, H = pil.size
            canvas = Image.new(pil.mode, (W, H), 0)
            # Use mask channel if available
            mask_rgba = region_scaled.split()[-1] if region_scaled.mode in ('RGBA','LA') else None
            canvas.paste(region_scaled, (offset_x, offset_y), mask=mask_rgba)

            if debug:
                draw = ImageDraw.Draw(canvas)
                draw.rectangle([x0_m, y0_m, x1_m, y1_m], outline='red')
                draw.rectangle([sx0_i, sy0_i, sx1_i, sy1_i], outline='blue')

            out_arr = np.array(canvas).astype(np.float32)/255.0
            outputs.append(out_arr)

        batch = np.stack(outputs, axis=0)
        if debug: print(f"Final batch shape: {batch.shape}")
        return (torch.from_numpy(batch),)



class ScaleHeightAroundBottomMidNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE", {"multiple": True}), "height_scale": ("FLOAT", {})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "scale_height"

    def scale_images(self, images, mask, scale_width):
        # 1) Mask bbox
        mask_pil = to_pil(mask).convert('L')
        mask_arr = np.array(mask_pil)
        ys, xs = np.nonzero(mask_arr > 0)
        if ys.size == 0:
            originals = []
            for img in images:
                pil = to_pil(img)
                arr = np.array(pil).astype(np.float32) / 255.0
                if arr.ndim == 2:
                    arr = arr[..., None]
                originals.append(arr)
            batch = np.stack(originals, axis=0)
            return (torch.from_numpy(batch),)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        ref_w, ref_h = x1 - x0, y1 - y0

        # 2) Convert to PIL
        pil_images = [to_pil(img) for img in images]

        # 3) First image non-black bbox
        arr0 = np.array(pil_images[0])
        non_black = np.any(arr0[..., :3] > 0, axis=-1)
        ys0, xs0 = np.nonzero(non_black)
        if ys0.size == 0:
            originals = []
            for pil in pil_images:
                arr = np.array(pil).astype(np.float32) / 255.0
                if arr.ndim == 2: arr = arr[..., None]
                originals.append(arr)
            batch = np.stack(originals, axis=0)
            return (torch.from_numpy(batch),)
        sy0, sy1 = ys0.min(), ys0.max()
        sx0, sx1 = xs0.min(), xs0.max()
        sk_w, sk_h = sx1 - sx0, sy1 - sy0

        # 4) Compute scales
        scale_y = ref_h / sk_h
        scale_x = (ref_w / sk_w) if scale_width else scale_y

        # 5) Inverse affine parameters for bottom-left alignment
        a = 1.0 / scale_x
        e = 1.0 / scale_y
        c = sx0 - x0 / scale_x  # align left of bbox
        f = sy1 - y1 / scale_y  # align bottom of bbox

        # 6) Apply affine transform
        transformed = []
        for pil in pil_images:
            W, H = pil.size
            img_t = pil.transform((W, H), Image.AFFINE, (a, 0.0, c, 0.0, e, f), resample=Image.BICUBIC)
            arr = np.array(img_t).astype(np.float32) / 255.0
            if arr.ndim == 2: arr = arr[..., None]
            transformed.append(arr)

        batch = np.stack(transformed, axis=0)
        return (torch.from_numpy(batch),)


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
            # because the skeleton won't start from top of head, so the ref_h should be scaled
            height_scale_factor = 0.8
            new_w, new_h = ref_w, int(ref_h * height_scale_factor)
            resized = pil.resize((new_w, new_h), Image.BILINEAR)
            paste_x, paste_y = x0, y0 + int((1 - height_scale_factor) * ref_h)

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
    "Neck":     (1, 0),
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
                    if scale == 1.0:
                        continue  # no change needed

                    # Head: scale around keypoint index 0
                    if region == "Head":
                        anchor = warped[0, :2]  # use keypoint index 0
                        coords = warped[list(idxs), :2]
                        warped[list(idxs), :2] = anchor + (coords - anchor) * scale
                    # Neck: scale neck bone, then translate all head points by that delta
                    elif region == "Neck":
                        anchor_idx, target_idx = idxs
                        anchor_pos = warped[anchor_idx, :2]
                        orig_target = warped[target_idx, :2]
                        direction = orig_target - anchor_pos
                        new_target = anchor_pos + direction * scale
                        delta = new_target - orig_target
                        warped[target_idx, :2] = new_target

                        # Move head region points by same delta
                        head_idxs = list(BODY_PART_INDEXES["Head"])
                        # don't move index 0 since it's already moved
                        head_idxs.remove(0)
                        warped[head_idxs, :2] = warped[head_idxs, :2] + delta
                    # RArm -> RForearm
                    elif region == "RArm":
                        a_i, t_i = idxs
                        anchor_pos = warped[a_i, :2]
                        orig_target = warped[t_i, :2].copy()
                        direction = orig_target - anchor_pos
                        new_target = anchor_pos + direction * scale
                        delta = new_target - orig_target
                        warped[t_i, :2] = new_target
                        child_ids = list(BODY_PART_INDEXES["RForearm"])
                        for idx in child_ids:
                            if idx not in idxs:
                                warped[idx, :2] += delta

                    # LArm -> LForearm
                    elif region == "LArm":
                        a_i, t_i = idxs
                        anchor_pos = warped[a_i, :2]
                        orig_target = warped[t_i, :2].copy()
                        direction = orig_target - anchor_pos
                        new_target = anchor_pos + direction * scale
                        delta = new_target - orig_target
                        warped[t_i, :2] = new_target
                        child_ids = list(BODY_PART_INDEXES["LForearm"])
                        for idx in child_ids:
                            if idx not in idxs:
                                warped[idx, :2] += delta

                    # RThigh -> RLeg
                    elif region == "RThigh":
                        a_i, t_i = idxs
                        anchor_pos = warped[a_i, :2]
                        orig_target = warped[t_i, :2].copy()
                        direction = orig_target - anchor_pos
                        new_target = anchor_pos + direction * scale
                        delta = new_target - orig_target
                        warped[t_i, :2] = new_target
                        child_ids = list(BODY_PART_INDEXES["RLeg"])
                        for idx in child_ids:
                            if idx not in idxs:
                                warped[idx, :2] += delta

                    # LThigh -> LLeg
                    elif region == "LThigh":
                        a_i, t_i = idxs
                        anchor_pos = warped[a_i, :2]
                        orig_target = warped[t_i, :2].copy()
                        direction = orig_target - anchor_pos
                        new_target = anchor_pos + direction * scale
                        delta = new_target - orig_target
                        warped[t_i, :2] = new_target
                        child_ids = list(BODY_PART_INDEXES["LLeg"])
                        for idx in child_ids:
                            if idx not in idxs:
                                warped[idx, :2] += delta

                    # SHOULDER: midpoint center, shift limbs separately
                    elif region == "Shoulder":
                        a_i, b_i = idxs
                        old_a = warped[a_i, :2].copy()
                        old_b = warped[b_i, :2].copy()
                        center = (old_a + old_b) / 2
                        rel_a = old_a - center
                        rel_b = old_b - center
                        new_a = center + rel_a * scale
                        new_b = center + rel_b * scale
                        warped[a_i, :2] = new_a
                        warped[b_i, :2] = new_b
                        delta_a = new_a - old_a
                        delta_b = new_b - old_b
                        for r in ["RArm","RForearm"]:
                            for idx in BODY_PART_INDEXES[r]:
                                if idx not in idxs:
                                    warped[idx, :2] += delta_a
                        for r in ["LArm","LForearm"]:
                            for idx in BODY_PART_INDEXES[r]:
                                if idx not in idxs:
                                    warped[idx, :2] += delta_b

                    # two-point bones: anchor at first, move second
                    elif len(idxs) == 2:
                        a_i, t_i = idxs
                        anchor_pos = warped[a_i, :2]
                        direction = warped[t_i, :2] - anchor_pos
                        warped[t_i, :2] = anchor_pos + direction * scale

                    # multi-point: midpoint-scaling
                    else:
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
        # Store original leg bone lengths to preserve
        orig_leg_lengths = {}
        # upper leg
        scale_factor_1 = 1.5
        # lower leg
        scale_factor_2 = 2
        for leg, (cp1, cp2, cp3) in cat_parts.items():
            if "hind" in leg:
                # Only resize hind leg parts
                L1 = np.linalg.norm(cat_kps[cp2,:2] - cat_kps[cp1,:2]) * scale_factor_1
                L2 = np.linalg.norm(cat_kps[cp3,:2] - cat_kps[cp2,:2]) * scale_factor_2
            else:
                L1 = np.linalg.norm(cat_kps[cp2,:2] - cat_kps[cp1,:2])
                L2 = np.linalg.norm(cat_kps[cp3,:2] - cat_kps[cp2,:2])
            orig_leg_lengths[leg] = (L1, L2)
        
        outputs = []
        for frame_idx, human in enumerate(human_poses):
            print(f"[DEBUG] Frame {frame_idx}")
            h_flat = human.get('people', [{}])[0].get('pose_keypoints_2d', [])
            human_kps = np.array(h_flat, dtype=float).reshape(-1,3)
            # Map hips proportionally
            xs = human_kps[:,0]; ys = human_kps[:,1]
            hxmin, hxmax = xs.min(), xs.max(); hymin, hymax = ys.min(), ys.max()
            hwidth = hxmax-hxmin if hxmax>hxmin else 1.0
            hheight= hymax-hymin if hymax>hymin else 1.0
            cat_xs = cat_kps[:,0]; cat_ys = cat_kps[:,1]
            cxmin, cxmax = cat_xs.min(), cat_xs.max(); cymin, cymax = cat_ys.min(), cat_ys.max()
            cwidth = cxmax-cxmin if cxmax>cxmin else 1.0
            cheight= cymax-cymin if cymax>cymin else 1.0
            # left hip
            hL = human_kps[11,:2]
            nx = (hL[0]-hxmin)/hwidth; ny = (hL[1]-hymin)/hheight
            cat_kps[11,:2] = np.array([cxmin+nx*cwidth, cymin+ny*cheight])
            # right hip
            hR = human_kps[8,:2]
            nx = (hR[0]-hxmin)/hwidth; ny = (hR[1]-hymin)/hheight
            cat_kps[14,:2] = np.array([cxmin+nx*cwidth, cymin+ny*cheight])
            # move cat neck normalized to human neck
            h_neck = human_kps[1,:2]
            nx = (h_neck[0]-hxmin)/hwidth; ny = (h_neck[1]-hymin)/hheight
            cat_kps[3,:2] = np.array([cxmin+nx*cwidth, cymin+ny*cheight])
            print(f"[DEBUG] Normalized cat neck idx3: {cat_kps[3,:2]}")
            # move cat eyes normalized to human eyes
            # human left eye idx14, right eye idx15
            h_leye = human_kps[14,:2]
            nx = (h_leye[0]-hxmin)/hwidth; ny = (h_leye[1]-hymin)/hheight
            cat_kps[0,:2] = np.array([cxmin+nx*cwidth, cymin+ny*cheight])
            h_reye = human_kps[15,:2]
            nx = (h_reye[0]-hxmin)/hwidth; ny = (h_reye[1]-hymin)/hheight
            cat_kps[1,:2] = np.array([cxmin+nx*cwidth, cymin+ny*cheight])
            # Create working copy for this frame
            new_cat = cat_kps.copy()

            # retarget legs
            for leg, (h1,h2,h3) in human_parts.items():
                cp1,cp2,cp3 = cat_parts[leg]
                origin = new_cat[cp1,:2]
                target = human_kps[h3,:2]
                L1,L2 = orig_leg_lengths[leg]
                # compute human bone dirs
                v1 = human_kps[h2,:2]-human_kps[h1,:2]; n1=np.linalg.norm(v1)
                v2 = human_kps[h3,:2]-human_kps[h2,:2]; n2=np.linalg.norm(v2)
                if n1<1e-6 or n2<1e-6:
                    print(f"[DEBUG] {leg} skipped due zero-length")
                    continue
                u1, u2 = v1/n1, v2/n2
                dot = np.clip(np.dot(u1,u2), -1.0,1.0); angle=math.acos(dot)
                sign = 1.0 if (u1[0]*u2[1]-u1[1]*u2[0])>=0 else -1.0
                elbow = origin + u1*L1
                paw = elbow + self.rotate2d(u1, sign*angle)*L2
                print(f"[DEBUG] {leg} elbow={elbow}, paw={paw}")
                new_cat[cp2,:2]=elbow; new_cat[cp3,:2]=paw

            # head
            h_nose=human_kps[0,:2]; h_neck=human_kps[1,:2]; v_head=h_nose-h_neck; n_head=np.linalg.norm(v_head)
            if n_head>1e-6:
                dir_h=v_head/n_head; idx_nose=2; idx_neck=3; Lh=np.linalg.norm(cat_kps[idx_nose,:2]-cat_kps[idx_neck,:2])
                new_cat[idx_nose,:2]=new_cat[idx_neck,:2]+dir_h*Lh; print(f"[DEBUG] head moved to {new_cat[idx_nose,:2]}")
            # torso
            h_mid=(human_kps[11,:2]+human_kps[12,:2])/2; v_torso=h_mid-h_neck; n_t=np.linalg.norm(v_torso)
            if n_t>1e-6:
                dir_t=v_torso/n_t; idx_tail=4; Lr=np.linalg.norm(cat_kps[idx_tail,:2]-cat_kps[idx_neck,:2])
                new_cat[idx_tail,:2]=new_cat[idx_neck,:2]+dir_t*Lr; print(f"[DEBUG] torso moved to {new_cat[idx_tail,:2]}")
            
            # adjust 5th keypoint mid of 12&15
            idx5=4; idx12=11; idx15=14
            mp=(new_cat[idx12,:2]+new_cat[idx15,:2])/2
            new_cat[idx5,:2]=mp; new_cat[idx5,2]=(new_cat[idx12,2]+new_cat[idx15,2])/2
            print(f"[DEBUG] 5th keypoint adjusted to {new_cat[idx5,:2]}")

            out={'animals':[new_cat.tolist()]}
            for k in('canvas_width','canvas_height'):
                if k in cat_dict: out[k]=cat_dict[k]
            outputs.append(out)

        return (outputs,)