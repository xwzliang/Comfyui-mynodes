import os,sys
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
WEB_DIRECTORY = "./web"
from .nodes import ScaleSkeletonsNode, CuteSkeletonNode, CatPoseRetargetNode, ScaleImagesToMaskNode, ScaleHeightAroundBottomMidNode
from .render_poses import RenderMultiplePeoplePoses, RenderMultipleAnimalPoses, DrawAnimalKeypoints, PoseJSONWrapper, RenderMultiplePeoplePosesForAnimal
from .unianimate_poses import MyWanVideoUniAnimateDWPoseDetector

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ScaleSkeletonsNode": ScaleSkeletonsNode,
    "CuteSkeletonNode": CuteSkeletonNode,
    "CatPoseRetargetNode": CatPoseRetargetNode,
    "RenderMultiplePeoplePoses": RenderMultiplePeoplePoses,
    "RenderMultipleAnimalPoses": RenderMultipleAnimalPoses,
    "DrawAnimalKeypoints": DrawAnimalKeypoints,
    "PoseJSONWrapper": PoseJSONWrapper,
    "ScaleImagesToMaskNode": ScaleImagesToMaskNode,
    "ScaleHeightAroundBottomMidNode": ScaleHeightAroundBottomMidNode,
    "RenderMultiplePeoplePosesForAnimal": RenderMultiplePeoplePosesForAnimal,
    "MyWanVideoUniAnimateDWPoseDetector": MyWanVideoUniAnimateDWPoseDetector,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ScaleSkeletonsNode": "Scale Skeletons Node",
    "CuteSkeletonNode": "Cute Skeleton Node",
    "CatPoseRetargetNode": "Cat Pose Retarget Node",
    "RenderMultiplePeoplePoses": "Render Multiple People Poses",
    "RenderMultipleAnimalPoses": "Render Multiple Animal Poses",
    "DrawAnimalKeypoints": "Draw animal pose keypoints",
    "PoseJSONWrapper": "Pose Json data to Keypoints",
    "ScaleImagesToMaskNode": "Scale Images to Mask Bbox",
    "ScaleHeightAroundBottomMidNode": "Scale Height Around Bottom Mid Node",
    "RenderMultiplePeoplePosesForAnimal": "Render Multiple People Poses for Animal",
    "MyWanVideoUniAnimateDWPoseDetector": "My Custom WanVideo UniAnimate DW Pose Detector",
}