import torch
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