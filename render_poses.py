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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "ControlNet Preprocessors/Pose Keypoint Postprocess"

    def render(self, kps):
        outputs = []
        for k in kps:
            _, poses, height, width = decode_json_as_poses(k)
            np_image = draw_animalposes(poses, height, width)
            out_t = torch.from_numpy(np_image.astype(np.float32) / 255.0)
            outputs.append(out_t)
        # instead of returning the raw list, stack into one tensor:
        stacked_outputs = torch.stack(outputs, dim=0)  # shape: (num_frames, H, W, 4)
        return (stacked_outputs,)