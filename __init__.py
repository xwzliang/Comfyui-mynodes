import os,sys
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
WEB_DIRECTORY = "./web"
from .nodes import ScaleSkeletonsNode, CuteSkeletonNode
from .render_poses import RenderMultiplePeoplePoses, RenderMultipleAnimalPoses

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ScaleSkeletonsNode": ScaleSkeletonsNode,
    "CuteSkeletonNode": CuteSkeletonNode,
    "RenderMultiplePeoplePoses": RenderMultiplePeoplePoses,
    "RenderMultipleAnimalPoses": RenderMultipleAnimalPoses,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ScaleSkeletonsNode": "Scale Skeletons Node",
    "CuteSkeletonNode": "Cute Skeleton Node",
    "RenderMultiplePeoplePoses": "Render Multiple People Poses",
    "RenderMultipleAnimalPoses": "Render Multiple Animal Poses",
}