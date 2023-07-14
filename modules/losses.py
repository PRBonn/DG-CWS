import math
import pdb
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as functional

# --------------------------------------- CROSS ENTROPY ---------------------------------------
  
class CrossEntropy(nn.Module):

  def __init__(self, weights: Optional[List] = None):
    super().__init__()

    if weights is not None:
      self.weights = torch.Tensor(weights)
    else:
      self.weights = None

  def forward(self, inputs: torch.Tensor, target: torch.Tensor, mode: str,
              mask_keep: Optional[torch.Tensor] = None) -> torch.Tensor:
    """ Compute cross entropy loss.

    Args:
        inputs (torch.Tensor): Unnormalized input tensor (logits) of shape [B x C x H x W]
        target (torch.Tensor): Ground-truth target tensor of shape [B x H x W]
        mode (str): train, val, or test
        mask_keep (Optional[torch.Tensor], optional): Mask of pixels of shape [B x H x W] which should be kept during loss computation (1 := keep, 0 := ignore). Defaults to None := keep all.

    Returns:
        torch.Tensor: loss value (scalar)
    """
    assert mode in ['train', 'val', 'test']

    if mask_keep is not None:
      target[mask_keep == False] = 0

    # get the number of classes and device
    batch_size, num_classes, height, width = inputs.shape
    input_device = inputs.device

    # convert logits to softmax probabilities
    probs = functional.softmax(inputs, dim=1)  # [N x n_classes x H x W]
    del inputs

    # apply one-hot encoding to ground truth annotations
    target_one_hot = to_one_hot(target, int(num_classes))  # [N x n_classes x H x W]
    target_one_hot = target_one_hot.bool()
    del target

    # prepare to ignore certain pixels which should not be considered during loss computation
    if mask_keep is None:
      # consider all pixels to compute the loss
      mask_keep = torch.ones((batch_size, 1, height, width), dtype=torch.bool, device=input_device) # [N x 1 x H x W]
    else:
      # get the dimension correctly
      mask_keep = mask_keep.unsqueeze(1)  # [N x 1 x H x W]
 
    # set ignore pixels to false
    target_one_hot = target_one_hot * mask_keep

    # gather the predicited probabilities of each ground truth category
    probs_gathered = probs[target_one_hot]  # M = N * (H * W) entries

    # make sure that probs are numerically stable when passed to log function: log(0) -> inf
    probs_gathered = torch.clip(probs_gathered, 1e-12, 1.0)

    # compute loss
    losses = -torch.log(probs_gathered)  # M = N * (H * W) entries
    del probs_gathered

    assert losses.shape[0] == torch.sum(mask_keep)
    del mask_keep
    
    # create weight matrix
    if self.weights is not None:
      if input_device != self.weights.device:
        self.weights = self.weights.to(input_device)

      weight_matrix = (target_one_hot.permute(0, 2, 3, 1) * self.weights).permute(0, 3, 1, 2)  # [N x n_classes x H x W]
      weights_gathered = weight_matrix[target_one_hot]  # M = N * (H * W) entries
      assert torch.all(weights_gathered > 0)

      # compute weighted loss for each prediction
      losses *= weights_gathered

    return torch.mean(losses)

def to_one_hot(tensor: torch.Tensor, n_classes: int) -> torch.Tensor:
  """ Convert tensor to its one hot encoded version.

  Props go to https://github.com/PRBonn/bonnetal/blob/master/train/common/onehot.py

  Args:
      tensor (torch.Tensor): ground truth tensor of shape [N x H x W]
      n_classes (int): number of classes

  Returns:
      torch.Tensor: one hot tensor of shape [N x n_classes x H x W]
  """
  if len(tensor.size()) == 1:
    b = tensor.size(0)
    if tensor.is_cuda:
      one_hot = torch.zeros(b, n_classes, device=torch.device('cuda')).scatter_(1, tensor.unsqueeze(1), 1)
    else:
      one_hot = torch.zeros(b, n_classes).scatter_(1, tensor.unsqueeze(1), 1)
  elif len(tensor.size()) == 2:
    n, b = tensor.size()
    if tensor.is_cuda:
      one_hot = torch.zeros(n, n_classes, b, device=torch.device('cuda')).scatter_(1, tensor.unsqueeze(1), 1)
    else:
      one_hot = torch.zeros(n, n_classes, b).scatter_(1, tensor.unsqueeze(1), 1)
  elif len(tensor.size()) == 3:
    n, h, w = tensor.size()
    if tensor.is_cuda:
      one_hot = torch.zeros(n, n_classes, h, w, device=torch.device('cuda')).scatter_(1, tensor.unsqueeze(1), 1)
    else:
      one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.unsqueeze(1), 1)
  return one_hot

def get_criterion(cfg) -> nn.Module:
  loss_name = cfg['train']['loss']

  if loss_name == 'xentropy':
    weights = cfg['train']['class_weights']

    return CrossEntropy(weights)
