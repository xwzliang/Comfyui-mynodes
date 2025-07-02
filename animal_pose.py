# import deeplabcut
video_path = "/workspace/ComfyUI/custom_nodes/Comfyui-mynodes/test/output.mp4"
superanimal_name = "superanimal_quadruped"
print(help(deeplabcut.video_inference_superanimal))
deeplabcut.video_inference_superanimal([video_path],
                                        superanimal_name,
                                        model_name="hrnet_w32",
                                        detector_name="fasterrcnn_resnet50_fpn_v2",
                                        video_adapt = False)