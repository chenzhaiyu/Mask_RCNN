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
DATASET_DIR = "/home/zhaiyu/Dataset/CMP_facade_DB_base"

# Configurations


class FacadeConfig(Config):
    """Configuration for training on the toy facade dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "facade"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 12  # background + 11 facade types

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    STEPS_PER_EPOCH = 200


config = FacadeConfig()
config.display()


class FacadeDataset(utils.Dataset):
    """Generates the facade dataset.
    """

    def load_facades(self, dataset_dir, subset):
        """Load a subset of the Facade dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes
        self.add_class("facade", 1, "1")
        self.add_class("facade", 2, "2")
        self.add_class("facade", 3, "3")
        self.add_class("facade", 4, "4")
        self.add_class("facade", 5, "5")
        self.add_class("facade", 6, "6")
        self.add_class("facade", 7, "7")
        self.add_class("facade", 8, "8")
        self.add_class("facade", 9, "9")
        self.add_class("facade", 10, "10")
        self.add_class("facade", 11, "11")
        self.add_class("facade", 12, "12")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        image_dir = os.path.join(dataset_dir, "images")
        mask_dir = os.path.join(dataset_dir, "masks")
        image_names = os.listdir(image_dir)

        for image_name in image_names:
            self.add_image("facade",
                           image_id=image_name,
                           path=os.path.join(image_dir, image_name),
                           mask_path=os.path.join(mask_dir, image_name[:-4] + ".png"))

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
        if image_info["source"] != "facade":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask_path = info["mask_path"]

        mask = (skimage.io.imread(mask_path, as_gray=True, plugin="pil") * 255).astype(np.uint8)

        # First detect how many little masks inside the image
        labels = measure.label(mask)
        masks_this_image = []
        class_ids = []
        for ch in range(1, np.max(labels) + 1):
            this_channel = (np.where(labels == ch, True, False))
            masks_this_image.append(this_channel)
            color_value = (np.max(mask * this_channel))
            if color_value == 12:
                class_id = 1
            elif color_value == 18:
                class_id = 2
            elif color_value == 36:
                class_id = 3
            elif color_value == 54:
                class_id = 4
            elif color_value == 79:
                class_id = 5
            elif color_value == 114:
                class_id = 6
            elif color_value == 140:
                class_id = 7
            elif color_value == 175:
                class_id = 8
            elif color_value == 200:
                class_id = 9
            elif color_value == 212:
                class_id = 10
            elif color_value == 224:
                class_id = 11
            elif color_value == 236:
                class_id = 12
            else:
                class_id = 1
                print("This color_value = {}".format(color_value))
                print("Shit happened! class_id == 1")
                return
            class_ids.append(class_id)

        masks_this_image = np.array(masks_this_image)
        # concatenated_masks = np.transpose(np.transpose(concatenated_masks, (2, 1, 0)), (1, 0, 2))
        if len(masks_this_image) == 0:
            print("No object mask here!")
            concatenated_masks = np.zeros((512, 512, 0))
        else:
            concatenated_masks = np.transpose(masks_this_image, (1, 2, 0))
        # class_ids = np.ones([np.max(labels)], dtype=np.int32)
        class_ids = np.array(class_ids, dtype=np.int32)
        return concatenated_masks.astype(np.bool), class_ids

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image


# Training dataset
dataset_train = FacadeDataset()
dataset_train.load_facades(DATASET_DIR, "train")
dataset_train.prepare()

# Validation dataset
dataset_val = FacadeDataset()
dataset_val.load_facades(DATASET_DIR, "val")
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
            epochs=15,  # used to be 2
            layers="all")

