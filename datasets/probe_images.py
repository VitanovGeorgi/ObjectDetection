"""
    This is done specifically for the dataset we have on hand. Here we also define the transforms for the images,
    were we to decide to have an augmented dataset.
"""

import os
from PIL import Image
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


def create_label_dict_resnet50(labelspath: str) -> dict:
    
    """
    Create a dictionary with the labels for the images. The labels are the bounding boxes for the objects in the images.
    The original labels are of type x1, y1, x2, y2. We convert them to x, y, width, height. Also for the resnet50 model,
    we need them of type "boxes" : torch.tensor([x1, y1, x2, y2], dtype=torch.float32), and "labels" : torch.tensor([1], dtype=torch.int64),
    i.e. the only thing we need to keep is those two keys. We keep them in the "bbox" key, as it is more convenient to extract them
    when training the model.
    """

    with open(labelspath) as f:
        original_json = json.load(f)

    output_dict = {}

    for element in original_json["images"]:

        _bbox = [
            annotation["bbox"] for annotation in original_json["annotations"] if annotation["image_id"] == element["id"]
        ][0]

        output_dict[
                element["file_name"]
            ] = {
                "height": element["height"],
                "id": element["id"],
                "width": element["width"],
                "bbox": [{
                    "boxes": torch.tensor(
                        [_bbox[0], _bbox[1], _bbox[0] + _bbox[2], _bbox[1] + _bbox[3]],
                        dtype=torch.float32
                    ),
                    "labels": torch.tensor([1], dtype=torch.int64)
                }],
            }


    return output_dict

def create_label_dict_x1_y1_x2_y2(labelspath: str) -> dict:
    """
    Create a dictionary with the labels for the images. The labels are the bounding boxes for the objects in the images.
    The original labels are of type x, y, width, height. We convert them to x1, y1, x2, y2.
    """

    with open(labelspath) as f:
        original_json = json.load(f)

    output_dict = {}

    for element in original_json["images"]:

        _bbox = [
            annotation["bbox"] for annotation in original_json["annotations"] if annotation["image_id"] == element["id"]
        ][0]

        output_dict[
                element["file_name"]
            ] = {
                "height": element["height"],
                "id": element["id"],
                "width": element["width"],
                "bbox": torch.tensor(
                    [_bbox[0], _bbox[1], _bbox[0] + _bbox[2], _bbox[1] + _bbox[3]],
                    dtype=torch.float32
                ),
            }


    return output_dict

def create_label_dict_x_y_w_h(labelspath: str) -> dict:

    """
    Create a dictionary with the labels for the images. The labels are the bounding boxes for the objects in the images.
    The original labels are of type x, y, width, height. We keep them as they are.
    """

    with open(labelspath) as f:
        original_json = json.load(f)

    output_dict = {}

    for element in original_json["images"]:
        output_dict[
                element["file_name"]
            ] = {
                "height": element["height"],
                "id": element["id"],
                "width": element["width"],
                "bbox": torch.tensor(
                    [
                        annotation["bbox"] for annotation in original_json["annotations"] if annotation["image_id"] == element["id"]
                    ][0], 
                    dtype=torch.float32
                ),
            }


    return output_dict

class ProbeImages_DatasetGenerator(Dataset):
    """
        Dataset generator for the probe images. It expects a root directory with the images, and a list of the images
    to be included. This is done. because the split of the images is done externally, and we want a different split
    at each initialization. Here we take care of labeling the images too.
    """
    def __init__(self, root_dir, images_list: list, labels_json_name: str, transform=None, model: str = "fasterrcnn_resnet50_fpn"):
        self.root_dir = root_dir
        self.images_path = os.listdir(root_dir)
        self.images = images_list
        self.labelspath = os.path.join(root_dir, labels_json_name) # we know it's only one json file
        if model == "fasterrcnn_resnet50_fpn":
            self.label_dict = create_label_dict_resnet50(self.labelspath)
        elif model == "vgg16":
            self.label_dict = create_label_dict_x1_y1_x2_y2(self.labelspath)
        elif model == "custom_cnn":
            self.label_dict = create_label_dict_x1_y1_x2_y2(self.labelspath)
        else:
            self.label_dict = create_label_dict_x_y_w_h(self.labelspath)
        self.transform = transform
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, self.images[idx], self.label_dict[self.images[idx]]


def train_val_test_split(
        data: list, 
        train_size: float = 0.7, 
        val_size: float = 0.2, 
        test_size: float = 0.1, 
        random_state: torch.Generator = 42,
        inference: bool = False
) -> tuple:
    """
    Given a list of data, split it into train, validation, and test sets. There's nothing to check whether they sum up to 1.
    """
    train, val_test = train_test_split(data, train_size=train_size, random_state=random_state)
    val, test = train_test_split(val_test, test_size=test_size / (val_size + test_size), random_state=random_state)
    if inference:
        return [], [], data
    return train, val, test

def train_test_split_probe_images(root_dir: str, random_state: torch.Generator = 42, inference: bool = False) -> tuple:
    """
    Given the root directory of the probe images, split them into train, validation, and test sets.
    We know that there are only the jpg images in the directory and the one json file which is the legend.

    Args:
        root_dir (str): The root directory of the probe images.
        random_state (torch.Generator): The random state for reproducibility.

    Returns:
        tuple: The train, validation, test sets.
    """
    
    probe_images_dir = os.listdir(root_dir)
    probe_images = [img for img in probe_images_dir if "jpg" in img]
    labels_images_json_name = [img for img in probe_images_dir if "json" in img][0]
    train_imgs, val_imgs, test_imgs = train_val_test_split(
        probe_images, 
        train_size=0.7, 
        val_size=0.2, 
        test_size=0.1, 
        random_state=random_state, 
        inference=inference
    )

    return train_imgs, val_imgs, test_imgs, labels_images_json_name




def get_probe_images_datasets(cfg: DictConfig, random_state: torch.Generator = 42) -> tuple:
    """
        Returns the datasets for the probe images. Here is where we can make different transforms to the images.
    """

    train_imgs, val_imgs, test_imgs, labels_images_json_name = train_test_split_probe_images(cfg.data.root_dir, random_state, inference=cfg.inference)

    

    if cfg.training.apply_augmentation:
        """ 
            The transformations were chosen randomly. They can be changed to anything else.
            The sole consideration was not to alter the geometry of the iamge, as we aren't changing the bounding boxes.
        """
        train_transform_aux = transforms.Compose([
            transforms.GaussianBlur(kernel_size=3),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        ])
        val_transform_aux = transforms.Compose([
            transforms.GaussianBlur(kernel_size=3),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        ])
        test_transform_aux = transforms.Compose([
            transforms.GaussianBlur(kernel_size=3),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        ])
    else:
        train_transform_aux = transforms.Compose([])
        val_transform_aux = transforms.Compose([])
        test_transform_aux = transforms.Compose([])


    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ] + train_transform_aux.transforms)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ] + val_transform_aux.transforms)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ] + test_transform_aux.transforms)



    if cfg.model.model == "vgg16" or cfg.model.model == "custom_cnn":
        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
            ] + train_transform.transforms
              + [transforms.Normalize(mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081])] # values chosen from MNIST greyscale
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
            ] + val_transform.transforms
              + [transforms.Normalize(mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081])]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
            ] + test_transform.transforms
              + [transforms.Normalize(mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081])]
        )



    # Datasets
    image_datasets = {
        "train": ProbeImages_DatasetGenerator(
            root_dir=cfg.data.root_dir,
            images_list=train_imgs, 
            labels_json_name=labels_images_json_name,
            transform=train_transform,
            model=cfg.model.model
        ),
        "val": ProbeImages_DatasetGenerator(
            root_dir=cfg.data.root_dir, 
            images_list=val_imgs, 
            labels_json_name=labels_images_json_name,
            transform=val_transform,
            model=cfg.model.model
        ),
        "test": ProbeImages_DatasetGenerator(
            root_dir=cfg.data.root_dir, 
            images_list=test_imgs, 
            labels_json_name=labels_images_json_name,
            transform=test_transform,
            model=cfg.model.model
        )
    }

    return image_datasets["train"], image_datasets["val"], image_datasets["test"]






