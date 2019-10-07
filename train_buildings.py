import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.curdir

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

import skimage.io
from skimage import measure

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Dataset Dir
DATASET_DIR = "/home/zhaiyu/Dataset/WHU Building Dataset"

# Configurations


class BuildingConfig(Config):
    """Configuration for training on the toy building dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "building"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + buildings

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512


config = BuildingConfig()
config.display()


class BuildingDataset(utils.Dataset):
    """Generates the building dataset.
    """

    def load_buildings(self, dataset_dir, subset):
        """Load a subset of the Building dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes
        self.add_class("building", 1, "building")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        image_dir = os.path.join(dataset_dir, "images")
        mask_dir = os.path.join(dataset_dir, "masks")
        image_names = os.listdir(image_dir)

        for image_name in image_names:
            self.add_image("building",
                           image_id=image_name,
                           path=os.path.join(image_dir, image_name),
                           width=512,
                           height=512,
                           mask_path=os.path.join(mask_dir, image_name))

    # def load_image(self, image_id):
    #     """Generate an image from the specs of the given image ID.
    #     Typically this function loads the image from a file
    #     """
    #     info = self.image_info[image_id]
    #     image = skimage.io.imread(info["path"], plugin='pil')
    #     return image

    def load_mask(self, image_id):
        """Generate instance masks given image ID.
        """
        # If not a ship dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "building":
            return super(self.__class__, self).load_mask(image_id)

        # Convert RLE Encoding to bitmap mask of shape [height, width, instance count]
        info = self.image_info[image_id]
        mask_path = info["mask_path"]
        shape = [info["height"], info["width"]]

        mask = skimage.io.imread(mask_path, plugin='pil')

        # First detect how many little masks inside the image
        labels = measure.label(mask)
        masks_this_image = []
        for ch in range(1, np.max(labels) + 1):
            this_channel = (np.where(labels == ch, True, False))
            masks_this_image.append(this_channel)

        masks_this_image = np.array(masks_this_image)
        # concatenated_masks = np.transpose(np.transpose(concatenated_masks, (2, 1, 0)), (1, 0, 2))
        if len(masks_this_image) == 0:
            print("No object mask here!")
            concatenated_masks = np.zeros((512, 512, 0))
        else:
            concatenated_masks = np.transpose(masks_this_image, (1, 2, 0))
        class_ids = np.ones([np.max(labels)], dtype=np.int32)

        return concatenated_masks.astype(np.bool), class_ids

    # def image_reference(self, image_id):
    #     """Return the path of the image."""
    #     info = self.image_info[image_id]
    #     if info["source"] == "building":
    #         return info["path"]
    #     else:
    #         super(self.__class__).image_reference(self, image_id)


# Training dataset
dataset_train = BuildingDataset()
dataset_train.load_buildings(DATASET_DIR, "train")
dataset_train.prepare()

# Validation dataset
dataset_val = BuildingDataset()
dataset_val.load_buildings(DATASET_DIR, "val")
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# Which weights to start with
init_with = "last"

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)

elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=4,  # used to be 2
            layers="all")

