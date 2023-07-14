"""
The variable names of the pretrained model at https://github.com/Eromera/erfnet/tree/master/trained_models
are different than ours.

Thus, we need to update the variables names according to our model structure in order to load the model.
Accordingly, this scripts updates the variables names and stores a new *.pth file that we can use as pretrained model.
"""

import argparse
import os
import pdb
from collections import OrderedDict
from typing import Dict

import torch

# define how to update the variable names
var_name_mapping = \
{
    "module.encoder.initial_block": "downsampler_block_01",
    "module.encoder.layers.0": "downsampler_block_02",
    "module.encoder.layers.1": "enc_non_bottleneck_1d_01",
    "module.encoder.layers.2": "enc_non_bottleneck_1d_02",
    "module.encoder.layers.3": "enc_non_bottleneck_1d_03",
    "module.encoder.layers.4": "enc_non_bottleneck_1d_04",
    "module.encoder.layers.5": "enc_non_bottleneck_1d_05",
    "module.encoder.layers.6": "downsampler_block_03",
    "module.encoder.layers.7": "enc_non_bottleneck_1d_06",
    "module.encoder.layers.8": "enc_non_bottleneck_1d_07",
    "module.encoder.layers.9": "enc_non_bottleneck_1d_08",
    "module.encoder.layers.10": "enc_non_bottleneck_1d_09",
    "module.encoder.layers.11": "enc_non_bottleneck_1d_10",
    "module.encoder.layers.12": "enc_non_bottleneck_1d_11",
    "module.encoder.layers.13": "enc_non_bottleneck_1d_12",
    "module.encoder.layers.14": "enc_non_bottleneck_1d_13",
    "module.decoder.layers.0": "upsampler_block_01",
    "module.decoder.layers.1": "dec_non_bottleneck_1d_01",
    "module.decoder.layers.2": "dec_non_bottleneck_1d_02",
    "module.decoder.layers.3": "upsampler_block_02",
    "module.decoder.layers.4": "dec_non_bottleneck_1d_03",
    "module.decoder.layers.5": "dec_non_bottleneck_1d_04",
    "module.decoder.output_conv": "segmentation_head.output_conv",
    "module.decoder.output_conv": "segmentation_head.output_conv"
}


def parse_args() -> Dict:
  parser = argparse.ArgumentParser()
  parser.add_argument('--fpath_model_pth', required=True, type=str, help="Path to original input .pth file.")
  parser.add_argument('--export_dir', required=True, help="Path to export directory to save the update .pth file")

  args = vars(parser.parse_args())

  if not os.path.exists(args['export_dir']):
    os.makedirs(args['export_dir'])

  return args


def load_state_dict(fpath_ckpt: str) -> OrderedDict:
  state_dict = torch.load(fpath_ckpt)

  return state_dict


def main():
  args = parse_args()

  # create new state dict with updated variable names
  prev_state_dict = load_state_dict(args['fpath_model_pth'])
  new_state_dict = dict()
  new_state_dict['state_dict'] = OrderedDict()
  for prev_var_name in prev_state_dict.keys():
    if 'module.decoder.output_conv' in prev_var_name:
      continue
    for name in var_name_mapping.keys():
      if 'layers' in prev_var_name:
        n = prev_var_name.split('.')[:4]
        n = '.'.join(n)
        if n == name:
          new_var_name = prev_var_name.replace(name, var_name_mapping[name])
          new_state_dict['state_dict'][new_var_name] = prev_state_dict[prev_var_name]
      elif prev_var_name.find(name) != -1:
        # create and assign updated var name
        new_var_name = prev_var_name.replace(name, var_name_mapping[name])
        new_state_dict['state_dict'][new_var_name] = prev_state_dict[prev_var_name]

  # dump new state dict to disk
  fpath_out = os.path.join(args['export_dir'], 'erfnet_pretrained_cvt.pth')
  torch.save(new_state_dict, fpath_out)


if __name__ == '__main__':
  main()
