import os
import pdb
import random
from typing import Callable, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from modules import style_transfers
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import datasets.common as common
from datasets.augmentations_color import get_color_augmentations
from datasets.augmentations_geometry import (GeometricDataAugmentation,
                                             get_geometric_augmentations)
from datasets.image_normalizer import ImageNormalizer, get_image_normalizer


class UAVDataset(Dataset):
  """ Represents the UAV Bonn dataset.

  The directory structure is as following:
  ├── annotations
  └── images
      └── rgb
  └── split.yaml
  """

  def __init__(self, path_to_dataset: str, 
                     filenames: List[str], mode: str, 
                     path_to_sparse_crop_dataset: str, 
                     path_to_sparse_weed_dataset: str, 
                     img_normalizer: ImageNormalizer,
                     augmentations_geometric: List[GeometricDataAugmentation], 
                     augmentations_color: List[Callable]):
    """ Get the path to all images and its corresponding annotations.

    Args:
        path_to_dataset (str): Path to dir that contains the images and annotations
        filenames (List[str]): List of filenames which are considered to be part of this dataset, e.g, [filename01.png, filename02.png, ...]
        path_to_sparse_crop_dataset (str): Path to dir that contains the dataset with sparsely annotated crops
        path_to_sparse_weed_dataset (str): Path to dir that contains the dataset with sparsely annotated weeds
        mode(str): train, val, or test
        img_normalizer (ImageNormalizer): Specifies how to normalize the input images
        augmentations_geometric (List[GeometricDataAugmentation]): Geometric data augmentations applied to the image and its annotations
        augmentations_color (List[Callable]): Color data augmentations applied to the image
    """

    assert os.path.exists(path_to_dataset), f"The path to the dataset does not exist: {path_to_dataset}."
    assert filenames, "The dataset is empty."

    super().__init__()

    self.filenames = filenames

    assert mode in ['train', 'val', 'test', 'predict']
    self.mode = mode

    self.img_normalizer = img_normalizer
    self.augmentations_geometric = augmentations_geometric
    self.augmentations_color = augmentations_color

    # --------- IMAGES WITH DENSE ANNOTATIONS ---------
    # get path to all RGB images
    self.path_to_images = os.path.join(path_to_dataset, "images", "rgb")
    assert os.path.exists(path_to_dataset)

    # get path to all annotations
    self.path_to_annos = os.path.join(path_to_dataset, "annotations")
    if mode != 'predict':
      assert os.path.exists(self.path_to_annos)

    # --------- CROP IMAGES WITH SPARSE ANNOTATIONS ---------
    # get path to all RGB images with sparse annotations
    self.path_to_sparse_crop_images = os.path.join(path_to_sparse_crop_dataset, "images", "rgb")
    self.path_to_sparse_crop_annos = os.path.join(path_to_sparse_crop_dataset, "annotations")
    self.fnames_of_sparsely_annotated_crop_images = common.get_img_fnames_in_dir(self.path_to_sparse_crop_images)
    assert self.fnames_of_sparsely_annotated_crop_images != 0

    # --------- WEED IMAGES WITH SPARSE ANNOTATIONS ---------
    # get path to all RGB images with sparse annotations
    self.path_to_sparse_weed_images = os.path.join(path_to_sparse_weed_dataset, "images", "rgb")
    self.path_to_sparse_weed_annos = os.path.join(path_to_sparse_weed_dataset, "annotations")
    self.fnames_of_sparsely_annotated_weed_images = common.get_img_fnames_in_dir(self.path_to_sparse_weed_images)
    assert self.fnames_of_sparsely_annotated_weed_images != 0

    # specify image transformations
    self.img_to_tensor = transforms.ToTensor()

  def get_train_item(self, idx: int) -> Dict:
    # --------- IMAGES WITH DENSE ANNOTATIONS ---------
    path_to_current_img = os.path.join(self.path_to_images, self.filenames[idx])
    img_pil = Image.open(path_to_current_img)
    img = self.img_to_tensor(img_pil)  # [C x H x W] with values in [0, 1]
    
    if random.random() > 0.25:
      for augmentor_color_fn in self.augmentations_color:
        img = augmentor_color_fn(img)
    img_clone = img.clone()
    
    path_to_current_anno = os.path.join(self.path_to_annos, self.filenames[idx])
    anno = np.array(Image.open(path_to_current_anno))  # dtype: int32
    if len(anno.shape) > 2:
        anno = anno[:, :, 0]
    anno = anno.astype(np.int64)
    anno = torch.Tensor(anno).type(torch.int64)  # [H x W]
    anno = anno.unsqueeze(0)  # [1 x H x W]
    anno_clone = anno.clone() 

    for augmentor_geometric in self.augmentations_geometric:
      img, anno = augmentor_geometric(img, anno)
    anno = anno.squeeze(0)  # [H x W]

    img_before_norm = img.clone()
    img = self.img_normalizer.normalize(img)
    
    # --------- CROP IMAGES WITH SPARSE ANNOTATIONS ---------
    fname_of_sparsely_annotated_crop_img = random.choice(self.fnames_of_sparsely_annotated_crop_images)

    path_crop_img_with_sparse_anno = os.path.join(self.path_to_sparse_crop_images, fname_of_sparsely_annotated_crop_img)
    img_pil_crop_with_sparse_anno = Image.open(path_crop_img_with_sparse_anno)
    img_crop_with_sparse_anno = self.img_to_tensor(img_pil_crop_with_sparse_anno)  # [C x H x W]
    
    if random.random() > 0.25:
      for augmentor_color_fn in self.augmentations_color:
        img_crop_with_sparse_anno = augmentor_color_fn(img_crop_with_sparse_anno)
    img_crops_clone = img_crop_with_sparse_anno.clone()

    path_sparse_crop_anno = os.path.join(self.path_to_sparse_crop_annos, fname_of_sparsely_annotated_crop_img)
    sparse_crop_anno = np.array(Image.open(path_sparse_crop_anno))  # dtype: uint8 of shape [H x W]
    sparse_crop_anno = sparse_crop_anno.astype(np.int64)
    sparse_crop_anno = torch.Tensor(sparse_crop_anno).type(torch.int64)  # [H x W]
    sparse_crop_anno = sparse_crop_anno.unsqueeze(0)  # [1 x H x W]
    crop_anno_clone = sparse_crop_anno.clone()

    for augmentor_geometric in self.augmentations_geometric:
      img_crop_with_sparse_anno, sparse_crop_anno = augmentor_geometric(img_crop_with_sparse_anno, sparse_crop_anno)
    sparse_crop_anno = sparse_crop_anno.squeeze(0)  # [H x W]
    
    img_crop_with_sparse_anno = self.img_normalizer.normalize(img_crop_with_sparse_anno)

    # --------- WEED IMAGES WITH SPARSE ANNOTATIONS ---------
    fname_of_sparsely_annotated_weed_img = random.choice(self.fnames_of_sparsely_annotated_weed_images)

    path_weed_img_with_sparse_anno = os.path.join(self.path_to_sparse_weed_images, fname_of_sparsely_annotated_weed_img)
    img_pil_weed_with_sparse_anno = Image.open(path_weed_img_with_sparse_anno)
    img_weed_with_sparse_anno = self.img_to_tensor(img_pil_weed_with_sparse_anno)  # [C x H x W]
  
    if random.random() > 0.25:
      for augmentor_color_fn in self.augmentations_color:
        img_weed_with_sparse_anno = augmentor_color_fn(img_weed_with_sparse_anno)
    img_weed_clone = img_weed_with_sparse_anno.clone()
    
    path_sparse_weed_anno = os.path.join(self.path_to_sparse_weed_annos, fname_of_sparsely_annotated_weed_img)
    sparse_weed_anno = np.array(Image.open(path_sparse_weed_anno))  # dtype: uint8 of shape [H x W]
    sparse_weed_anno = sparse_weed_anno.astype(np.int64)
    sparse_weed_anno = torch.Tensor(sparse_weed_anno).type(torch.int64)  # [H x W]
    sparse_weed_anno = sparse_weed_anno.unsqueeze(0)  # [1 x H x W]
    weed_anno_clone = sparse_weed_anno.clone()

    for augmentor_geometric in self.augmentations_geometric:
      img_weed_with_sparse_anno, sparse_weed_anno = augmentor_geometric(img_weed_with_sparse_anno, sparse_weed_anno)
    sparse_weed_anno = sparse_weed_anno.squeeze(0)  # [H x W]
    
    img_weed_with_sparse_anno = self.img_normalizer.normalize(img_weed_with_sparse_anno)

    assert (not img.isnan().any()), f"Invalid input image: {path_to_current_img}"
    assert (not img_crop_with_sparse_anno.isnan().any()), f"Invalid input image: {path_crop_img_with_sparse_anno}"
    assert (not img_weed_with_sparse_anno.isnan().any()), f"Invalid input image: {path_weed_img_with_sparse_anno}"
    assert (not anno.isnan().any()), f"Invalid input anno: {path_to_current_anno}"
    assert (not sparse_crop_anno.isnan().any()), f"Invalid input image: {path_sparse_crop_anno}"
    assert (not sparse_weed_anno.isnan().any()), f"Invalid input image: {path_sparse_weed_anno}"
    
    # Transfer global style of crops
    img_global_style_crops = style_transfers.transfer_global_statistics(img_crops_clone.unsqueeze(0), img_clone.unsqueeze(0))
    img_global_style_crops = img_global_style_crops.squeeze(0)
    
    anno_global_style_crops = anno_clone.clone()
    for augmentor_geometric in self.augmentations_geometric:
      img_global_style_crops, anno_global_style_crops = augmentor_geometric(img_global_style_crops, anno_global_style_crops)
    anno_global_style_crops = anno_global_style_crops.squeeze(0)

    img_global_style_crops = self.img_normalizer.normalize(img_global_style_crops)

    # Transfer global style of weeds
    img_global_style_weeds = style_transfers.transfer_global_statistics(img_weed_clone.unsqueeze(0), img_clone.unsqueeze(0))
    img_global_style_weeds = img_global_style_weeds.squeeze(0)
    
    anno_global_style_weeds = anno_clone.clone()
    for augmentor_geometric in self.augmentations_geometric:
      img_global_style_weeds, anno_global_style_weeds = augmentor_geometric(img_global_style_weeds, anno_global_style_weeds)
    anno_global_style_weeds = anno_global_style_weeds.squeeze(0)

    img_global_style_weeds = self.img_normalizer.normalize(img_global_style_weeds)

    return {'input_image_before_norm': img_before_norm, 
            'input_image': img, 
            'anno': anno, 
            'fname': self.filenames[idx], 
            'img_crop_with_sparse_anno': img_crop_with_sparse_anno, 
            'sparse_crop_anno': sparse_crop_anno, 
            'fname_sparse_crops': fname_of_sparsely_annotated_crop_img, 
            'img_weed_with_sparse_anno': img_weed_with_sparse_anno, 
            'sparse_weed_anno': sparse_weed_anno,
            'fname_sparse_weeds': fname_of_sparsely_annotated_weed_img,
            'img_global_style_crops': img_global_style_crops,
            'anno_global_style_crops': anno_global_style_crops,
            'img_global_style_weeds': img_global_style_weeds,
            'anno_global_style_weeds': anno_global_style_weeds}

  def get_val_or_test_item(self, idx:int) -> Dict:
    path_to_current_img = os.path.join(self.path_to_images, self.filenames[idx])
    img_pil = Image.open(path_to_current_img)
    img = self.img_to_tensor(img_pil)  # [C x H x W]

    path_to_current_anno = os.path.join(self.path_to_annos, self.filenames[idx])
    anno = np.array(Image.open(path_to_current_anno))  # dtype: int32
    if len(anno.shape) > 2:
        anno = anno[:, :, 0]
    anno = anno.astype(np.int64)
    anno[anno == 10] = 1
    anno[anno == 20] = 2
    anno = torch.Tensor(anno).type(torch.int64)  # [H x W]
    anno = anno.unsqueeze(0)  # [1 x H x W]

    for augmentor_geometric in self.augmentations_geometric:
      img, anno = augmentor_geometric(img, anno)
    anno = anno.squeeze(0)  # [H x W]

    img_before_norm = img.clone()
    img = self.img_normalizer.normalize(img)

    assert (not img.isnan().any()), f"Invalid input image: {path_to_current_img}"
    assert (not anno.isnan().any()), f"Invalid input anno: {path_to_current_anno}"

    return {'input_image_before_norm': img_before_norm, 'input_image': img, 'anno': anno, 'fname': self.filenames[idx]}

  def get_predict_item(self, idx: int) -> Dict:
    path_to_current_img = os.path.join(self.path_to_images, self.filenames[idx])
    img_pil = Image.open(path_to_current_img)
    img = self.img_to_tensor(img_pil)  # [C x H x W]

    # create dummy anno to make everything conistent
    anno = torch.zeros((1, img.shape[1], img.shape[2]), dtype=torch.int64)

    for augmentor_geometric in self.augmentations_geometric:
      img, anno = augmentor_geometric(img, anno)
    anno = anno.squeeze(0)  # [H x W]

    img_before_norm = img.clone()
    img = self.img_normalizer.normalize(img)

    return {'input_image_before_norm': img_before_norm, 'input_image': img, 'anno': anno, 'fname': self.filenames[idx], 'img_with_sparse_anno': torch.zeros_like(img), 'sparse_anno': torch.zeros_like(anno)}

  def __getitem__(self, idx: int):
    if self.mode == 'train':
      items = self.get_train_item(idx)
      return items
    if (self.mode == 'val') or (self.mode == 'test'):
      items = self.get_val_or_test_item(idx)
      return items
    if self.mode == 'predict':
      items = self.get_predict_item(idx)
      return items
    
  def __len__(self) -> int:
    return len(self.filenames)


class UAVBonnDataModule(pl.LightningDataModule):
  """ Encapsulates all the steps needed to process data from UAV Bonn.
  """

  def __init__(self, cfg: Dict):
    super().__init__()

    self.cfg = cfg

  def setup(self, stage: Optional[str] = None):
    """ Data operations we perform on every GPU.

    Here we define the how to split the dataset.

    Args:
        stage (Optional[str], optional): _description_. Defaults to None.
    """
    path_to_dataset = self.cfg['data']['path_to_dataset']
    path_to_sparse_crop_dataset = self.cfg['data']['path_to_sparse_crop_dataset']
    path_to_sparse_weed_dataset = self.cfg['data']['path_to_sparse_weed_dataset']
    image_normalizer = get_image_normalizer(self.cfg)

    path_to_split_file = os.path.join(self.cfg['data']['path_to_dataset'], 'split.yaml')
    if stage != "predict":
      if self.cfg['data']['check_data_split']:
        split_file_is_valid = common.check_split_file(path_to_split_file)
        assert split_file_is_valid, "The train, val, and test splits 'split.yaml' are not mutually exclusive."

      with open(path_to_split_file) as istream:
        split_info = yaml.safe_load(istream)

      if (stage == 'fit') or (stage == 'validate') or (stage is None):
        # ----------- TRAIN -----------
        train_filenames = split_info['train']
        train_filenames.sort()

        if self.cfg['train']['dataset_size'] is not None:
          train_filenames = train_filenames[:self.cfg['train']['dataset_size']]

        train_augmentations_geometric = get_geometric_augmentations(self.cfg, 'train')
        train_augmentations_color = get_color_augmentations(self.cfg, 'train')
        self._uav_train = UAVDataset(
            path_to_dataset,
            train_filenames,
            mode='train',
            path_to_sparse_crop_dataset=path_to_sparse_crop_dataset,
            path_to_sparse_weed_dataset=path_to_sparse_weed_dataset,
            img_normalizer=image_normalizer,
            augmentations_geometric=train_augmentations_geometric,
            augmentations_color=train_augmentations_color)

        # ----------- VAL -----------
        val_filenames = split_info['valid']
        val_filenames.sort()

        if self.cfg['val']['dataset_size'] is not None:
          val_filenames = val_filenames[:self.cfg['val']['dataset_size']]

        val_augmentations_geometric = get_geometric_augmentations(self.cfg, 'val')
        self._uav_val = UAVDataset(
            path_to_dataset,
            val_filenames,
            mode='val',
            path_to_sparse_crop_dataset=path_to_sparse_crop_dataset,
            path_to_sparse_weed_dataset=path_to_sparse_weed_dataset,
            img_normalizer=image_normalizer,
            augmentations_geometric=val_augmentations_geometric,
            augmentations_color=[])

      if stage == 'test' or stage is None:
        # ----------- TEST -----------
        test_filenames = split_info['test']
        test_filenames.sort()

        if self.cfg['test']['dataset_size'] is not None:
          test_filenames = test_filenames[:self.cfg['test']['dataset_size']]

        test_augmentations_geometric = get_geometric_augmentations(self.cfg, 'test')
        self._uav_test = UAVDataset(
            path_to_dataset,
            test_filenames,
            mode='test',
            path_to_sparse_crop_dataset=path_to_sparse_crop_dataset,
            path_to_sparse_weed_dataset=path_to_sparse_weed_dataset,
            img_normalizer=image_normalizer,
            augmentations_geometric=test_augmentations_geometric,
            augmentations_color=[])

    if stage == "predict":
      predict_augmentations_geometric = get_geometric_augmentations(self.cfg, 'predict')

      path_to_images = os.path.join(path_to_dataset, 'images', 'rgb')
      predict_filenames = common.get_img_fnames_in_dir(path_to_images)

      self._uav_predict = UAVDataset(
          path_to_dataset,
          predict_filenames,
          mode='predict',
          path_to_sparse_crop_dataset=path_to_sparse_crop_dataset,
          path_to_sparse_weed_dataset=path_to_sparse_weed_dataset,
          img_normalizer=image_normalizer,
          augmentations_geometric=predict_augmentations_geometric,
          augmentations_color=[])

  def train_dataloader(self) -> DataLoader:
    # Return DataLoader for Training Data here
    shuffle: bool = self.cfg['train']['shuffle']
    batch_size: int = self.cfg['train']['batch_size']
    n_workers: int = self.cfg['data']['num_workers']

    loader = DataLoader(self._uav_train, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers, drop_last=True, pin_memory=True)

    return loader

  def val_dataloader(self) -> DataLoader:
    batch_size: int = self.cfg['val']['batch_size']
    n_workers: int = self.cfg['data']['num_workers']

    loader = DataLoader(self._uav_val, batch_size=batch_size, num_workers=n_workers, shuffle=False, drop_last=True, pin_memory=True)

    return loader

  def test_dataloader(self) -> DataLoader:
    batch_size: int = self.cfg['test']['batch_size']
    n_workers: int = self.cfg['data']['num_workers']

    loader = DataLoader(self._uav_test, batch_size=batch_size, num_workers=n_workers, shuffle=False, pin_memory=True)

    return loader

  def predict_dataloader(self):
    batch_size: int = self.cfg['predict']['batch_size']
    n_workers: int = self.cfg['data']['num_workers']

    loader = DataLoader(self._uav_predict, batch_size=batch_size, num_workers=n_workers, shuffle=False, pin_memory=True)

    return loader
