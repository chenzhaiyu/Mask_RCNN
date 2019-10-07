import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
SAVE_DIR = os.path.abspath("./results/building")
# SAVE_TYPE = "instance"
SAVE_TYPE = "mask"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import samples.coco.coco as coco

import cv2

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = "models/16_mAP_0.8625_pre_0.8761_rec_0.8856_60.h5"
# COCO_MODEL_PATH = "logs/building20190928T1841/mask_rcnn_building_0004.h5"
# COCO_MODEL_PATH = "/home/zhaiyu/Dataset/mask_rcnn_coco.h5"
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images/new")
IMAGE_DIR = "/home/zhaiyu/Dataset/WHU Building Dataset/test/images"
# Configuration


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1


config = InferenceConfig()
config.display()

# Create Model and Load Trained Weights

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
# class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                'bus', 'train', 'truck', 'boat', 'traffic light',
#                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                'teddy bear', 'hair drier', 'toothbrush']

class_names = ['BG', 'buildings']

# Run Object Detection

file_names = next(os.walk(IMAGE_DIR))[2]

for file_name in file_names:
    if not file_name.endswith("tif"):
        continue
    if file_name.startswith("."):
        continue

    image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

    # Skip this image if found in SAVE_DIR
    output_mask_names = next(os.walk(SAVE_DIR))[2]
    if file_name in output_mask_names:
        print("already found " + file_name + " in directory")
        continue

    # Run detection
    results = model.detect([image], verbose=1)
    r = results[0]

    assert SAVE_TYPE in ["instance", "mask"]

    if SAVE_TYPE == "instance":
        output = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'],
                                                 is_display=False)

    elif SAVE_TYPE == "mask":
        # Number of boxes
        N = results[0]['rois'].shape[0]
        # Masks with 90% or higher confidence
        masks = results[0]['masks']  # can also get mask probability by 'probs'
        # Boxes
        boxes = results[0]['rois']

        if masks is None or boxes is None:
            UserWarning("No masks or boxes")
            continue

        overlay_mask = np.zeros((masks.shape[:-1]), dtype=bool)
        for i in range(masks.shape[-1]):
            overlay_mask += masks[:, :, i]
        overlay_mask = overlay_mask.astype("uint8")
        output = overlay_mask * 255

    cv2.imwrite(os.path.join(SAVE_DIR, file_name), output.astype(np.uint8))


# # Visualize results
# r = results[0]
# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                             class_names, r['scores'])
