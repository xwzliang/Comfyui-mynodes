import numpy as np
import torch
from PIL import Image

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

