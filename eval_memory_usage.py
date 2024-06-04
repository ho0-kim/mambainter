import os

import torch

from model.propainter import InpaintGenerator as ProPainter
from model.mambainter import InpaintGenerator as Mambainter
from utils.download_util import load_file_from_url
from model.misc import get_device

import torchsummaryX
import torchinfo

def model_summarize(model, examples):
  model.eval()

  # Run model
  if isinstance(model, ProPainter):
    pred_img = model(examples[0], examples[1], examples[2], examples[3], examples[4])
  elif isinstance(model, Mambainter):
    pred_img = model(examples[0], examples[1], examples[2], examples[3], examples[4], examples[5], examples[6])

  # Summarize model
  if isinstance(model, ProPainter):
    torchinfo.summary(model, input_data=examples[:5])
  elif isinstance(model, Mambainter):
    torchinfo.summary(model, input_data=examples)


if __name__ == '__main__':
  # Load model
  device = get_device()

#   model = ProPainter().to(device)
  model = Mambainter().to(device)

  # Examples
  n = 5
  l_t = 3
  index_list = [i for i in range(n)]
  index_mask = [False, True, True, True, False]#[True for i in range(n)]
  selected_imgs = torch.rand((1, n, 3, 240, 432), device=device)
  selected_pred_flows_bi = torch.rand((2, 1, l_t-1, 2, 240, 432), device=device)
  selected_masks = torch.rand((1, n, 1, 240, 432), device=device)
  selected_update_masks = torch.rand((1, n, 1, 240, 432), device=device)

  examples = [selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t, index_list, index_mask]

  model_summarize(model, examples)

  # torchsummaryX.summary(model, 
  #     selected_imgs, 
  #     selected_pred_flows_bi, 
  #     selected_masks, 
  #     selected_update_masks, 
  #     l_t, 
  #     index_list, 
  #     index_mask)