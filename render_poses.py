import torch
import json
import cv2
import numpy as np
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.dwpose import decode_json_as_poses, draw_poses, draw_animalposes
# from custom_nodes.comfyui_controlnet_aux.node_wrappers.pose_keypoint_postprocess import numpy2torch

class RenderMultiplePeoplePoses:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "kps": ("POSE_KEYPOINT",),
                "render_body": ("BOOLEAN", {"default": True}),
                "render_hand": ("BOOLEAN", {"default": True}),
                "render_face": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "ControlNet Preprocessors/Pose Keypoint Postprocess"

    def render(self, kps, render_body, render_hand, render_face):
        outputs = []
        for k in kps:
            poses, _, height, width = decode_json_as_poses(k)
            np_image = draw_poses(
                poses,
                height,
                width,
                render_body,
                render_hand,
                render_face,
            )
            out_t = torch.from_numpy(np_image.astype(np.float32) / 255.0)
            outputs.append(out_t)
        # instead of returning the raw list, stack into one tensor:
        stacked_outputs = torch.stack(outputs, dim=0)  # shape: (num_frames, H, W, 4)
        return (stacked_outputs,)

class RenderMultipleAnimalPoses:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "kps": ("POSE_KEYPOINT",),
                # any confidence ≥ this value will be clamped to 1.0
                "threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "ControlNet Preprocessors/Pose Keypoint Postprocess"

    def render(self, kps, threshold):
        def _flatten(pose):
            # convert nested [[x,y,c],...] into [x,y,c,x,y,c,...]
            return [v for triplet in pose for v in triplet]

        outputs = []
        for k in kps:
            # shallow copy to avoid mutating the original dict
            k2 = dict(k)

            # adjust animal confidences before decoding
            if "animals" in k2:
                new_animals = []
                for a in k2["animals"]:
                    # ensure we have a flat list of floats
                    flat = _flatten(a) if isinstance(a[0], (list, tuple)) else list(a)
                    # clamp any confidences above the threshold up to 1.0
                    for i in range(2, len(flat), 3):  # every 3rd element is the confidence c
                        if flat[i] >= threshold:
                            flat[i] = 1.0
                    new_animals.append(flat)
                k2["animals"] = new_animals

            # decode keypoints into poses
            _, animal_poses, H, W = decode_json_as_poses(k2)

            # render poses onto a blank canvas
            img = draw_animalposes(animal_poses, H, W)
            # convert to a torch tensor in [0..1]
            tensor = torch.from_numpy(img.astype("float32") / 255.0)
            outputs.append(tensor)

        # stack into a batch (num_frames, H, W, 3)
        return (torch.stack(outputs, dim=0),)


class DrawAnimalKeypoints:
    """
    ComfyUI custom node: DrawAnimalKeypoints
    Draws AP-10k animal keypoints on a generated blank canvas.

    Inputs:
    - cat_pose: POSE_KEYPOINT (Python list from pose estimator result, as shown)
    - threshold: FLOAT, confidence threshold

    Output:
    - IMAGE: torch.Tensor ([C,H,W], float32 in [0,1]) with keypoints overlay
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # cat_pose is passed directly as the Python object from POSE_KEYPOINT
                "cat_pose": ("POSE_KEYPOINT",),
                "threshold": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw_keypoints"
    CATEGORY = "Image Processing"

    def draw_keypoints(self, cat_pose, threshold=0.6):
        # cat_pose is expected as a Python list/dict structure, not a JSON string
        data = cat_pose
        # Ensure the structure is as expected: a list with one dict
        if not isinstance(data, list) or not data:
            return (None,)
        entry = data[0]

        # Extract canvas dimensions
        canvas_w = entry.get("canvas_width")
        canvas_h = entry.get("canvas_height")
        if not isinstance(canvas_w, int) or not isinstance(canvas_h, int):
            return (None,)

        # Extract first animal's keypoints
        animals = entry.get("animals")
        if not isinstance(animals, list) or not animals or not isinstance(animals[0], list):
            return (None,)
        keypoints = animals[0]

        # Create blank white canvas (HWC uint8)
        img = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        # Draw keypoints above threshold
        for idx, kp in enumerate(keypoints):
            if not isinstance(kp, (list, tuple)) or len(kp) < 3:
                continue
            x, y, score = kp
            if score < threshold:
                continue
            px, py = int(x), int(y)
            if 0 <= px < canvas_w and 0 <= py < canvas_h:
                cv2.circle(img, (px, py), 5, (0, 255, 0), -1)
                cv2.putText(img, str(idx+1), (px + 6, py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Convert to torch.Tensor [C,H,W], float32 in [0,1]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        return (tensor,)


class PoseJSONWrapper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pose_json": ("STRING", {"default": "[]"}),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    FUNCTION = "wrap_pose"
    CATEGORY = "Custom/POSE_KEYPOINT"

    def wrap_pose(self, image, pose_json):
        # 1) parse incoming JSON string
        pose_list = json.loads(pose_json)

        # 2) inspect tensor shape and pull H, W
        shape = image.shape  # torch.Size, e.g. [1,1856,1024,3]
        if len(shape) == 4:
            _, h, w, _ = shape
        elif len(shape) == 3:
            h, w, _ = shape
        else:
            raise ValueError(f"Unexpected image shape: {shape}")

        # 3) wrap into your target schema
        result = [{
            "version": "ap10k",
            "animals": [pose_list],
            "canvas_width": w,
            "canvas_height": h
        }]

        return (result,)