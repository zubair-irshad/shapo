import os
import copy

os.environ['PYTHONHASHSEED'] = str(1)
from importlib.machinery import SourceFileLoader
import random
random.seed(123456)
import numpy as np
np.random.seed(123456)
import torch
import wandb
import pytorch_lightning as pl
import torch.distributed as dist

from simnet.lib.net import common
from simnet.lib.net.dataset import extract_left_numpy_img
from simnet.lib.net.functions.learning_rate import lambda_learning_rate_poly, lambda_warmup

from simnet.lib.net.post_processing.losses import compute_depth_loss, DisparityLoss, compute_seg_loss, compute_pose_shape_loss
from simnet.lib.net.post_processing.visualization import get_seg_visualization_img, get_depth_visualization_img

_GPU_TO_USE = 0

class PanopticModel(pl.LightningModule):
  def __init__(
      self, hparams, epochs=None, train_dataset=None, eval_metric=None, preprocess_func=None
  ):
    super().__init__()

    self.hyperparams = hparams
    self.epochs = epochs
    self.train_dataset = train_dataset

    self.model = common.get_model(hparams)
    self.eval_metrics = eval_metric
    self.preprocess_func = preprocess_func
    self.save_hyperparameters()
    _MAX_DISP = 128
    self.disp_loss = DisparityLoss(_MAX_DISP, False)

  def forward(self, image):
    # seg_output, depth_output, small_depth_output, pose_output = self.model(
    #     image
    # )
    seg_output, depth_output, small_disp_output, heatmap_output, shape_emb_output, appearance_emb_output, abs_pose_output = self.model(image)

    #print("seg_output, depth_output, small_disp_output, heatmap_output, shape_emb_output, appearance_emb_output, abs_pose_output", seg_output.shape, depth_output.shape, small_disp_output.shape, heatmap_output.shape, shape_emb_output.shape, appearance_emb_output.shape, abs_pose_output.shape)
    return seg_output, depth_output, small_disp_output, heatmap_output, shape_emb_output, appearance_emb_output, abs_pose_output
  
  # def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, second_order_closure=None):
  #   super().optimizer_step(epoch_nb, batch_nb, optimizer, optimizer_i, second_order_closure)
  #   if batch_nb == 0:
  #     for param_group in optimizer.param_groups:
  #       learning_rate = param_group['lr']

  def optimizer_step(self, *args, **kwargs):
      super().optimizer_step(*args, **kwargs)

  def training_step(self, batch, batch_idx):
    # image, seg_target, depth_target, pose_targets, _, _ = batch

    image, seg_target, depth_target, heatmap_target, shape_emb_target, appearance_emb_target, abs_pose_target = batch
    # seg_output, depth_output, small_depth_output, pose_outputs = self.forward(
    #     image
    # )

    seg_output, depth_output, small_disp_output, heatmap_output, shape_emb_output, appearance_emb_output, abs_pose_output = self.forward(image)
    log = {}
    prefix = 'train'
    
    # loss = depth_output.compute_loss(copy.deepcopy(depth_target), log, f'{prefix}_detailed/loss/refined_disp')

    loss = compute_depth_loss(self.disp_loss, depth_output, copy.deepcopy(depth_target), log, f'{prefix}_detailed/loss/train_cost_volume_disp', self.hyperparams)
    # if self.hyperparams.depth_output is None:
    #   loss = loss + small_depth_output.compute_loss(depth_target, log, f'{prefix}_detailed/loss/train_cost_volume_disp')

    if self.hyperparams.frozen_stereo_checkpoint is None:
      loss = loss + compute_depth_loss(self.disp_loss, small_disp_output, depth_target, log, f'{prefix}_detailed/loss/train_cost_volume_disp', self.hyperparams)

    loss = loss + compute_seg_loss(seg_output, seg_target, log, f'{prefix}_detailed/loss/seg', self.hyperparams)
    
    # loss = loss + seg_output.compute_loss(seg_target, log, f'{prefix}_detailed/loss/seg')
    # if pose_targets[0] is not None:
      # loss = loss + pose_outputs.compute_loss(pose_targets, log, f'{prefix}_detailed/pose')
    loss = loss + compute_pose_shape_loss(self, heatmap_target, shape_emb_target, appearance_emb_target, abs_pose_target, heatmap_output, shape_emb_output, appearance_emb_output, abs_pose_output, log, f'{prefix}_detailed/pose', self.hyperparams)
    
    log['train/loss/total'] = loss
    logger = self.logger.experiment
    logger.log(log)

    #TODO, fix visualization
    rank = dist.get_rank()
    if (batch_idx % 200) == 0 and rank ==0:
      with torch.no_grad():
        llog = {}
        prefix = 'train'
        left_image_np = extract_left_numpy_img(image[0])
       
        seg_pred_vis = get_seg_visualization_img(left_image_np, seg_output)
        llog[f'{prefix}/seg'] = wandb.Image(seg_pred_vis, caption=prefix)

        depth_vis = get_depth_visualization_img(left_image_np, depth_output)
        llog[f'{prefix}/disparity'] = wandb.Image(depth_vis, caption=prefix)

        small_depth_vis = get_depth_visualization_img(left_image_np, small_disp_output)
        llog[f'{prefix}/small_disparity'] = wandb.Image(small_depth_vis, caption=prefix)
        logger.log(llog)

    # if (batch_idx % 1000) == 0 and :
    #   with torch.no_grad():
    #     llog = {}
    #     prefix = 'train'
    #     left_image_np = extract_left_numpy_img(image[0])
    #     seg_pred_vis = seg_output.get_visualization_img(np.copy(left_image_np))
    #     llog[f'{prefix}/seg'] = wandb.Image(seg_pred_vis, caption=prefix)
    #     depth_vis = depth_output.get_visualization_img(np.copy(left_image_np))
    #     llog[f'{prefix}/disparity'] = wandb.Image(depth_vis, caption=prefix)
    #     small_depth_vis = small_depth_output.get_visualization_img(np.copy(left_image_np))
    #     llog[f'{prefix}/small_disparity'] = wandb.Image(small_depth_vis, caption=prefix)
    #     logger.log(llog)
    return {'loss': loss, 'log': log}

  def validation_step(self, batch, batch_idx):
    # image, seg_target, depth_target, pose_targets, _, scene_name = batch
    # # dt = [di.depth_pred.cuda().unsqueeze(0) for di in depth_target]
    # # dt = [di.depth_pred.to(image.device).unsqueeze(0) for di in depth_target]

    # dt = [di.depth_pred.to(self.device).unsqueeze(0) for di in depth_target]
    # dt = torch.stack(dt)
    # real_image = torch.cat([image[:,:3,:,:], dt], dim=1)
    # seg_output, depth_output, small_depth_output, pose_outputs = self.forward(
    #     real_image
    # )

    image, seg_target, depth_target, heatmap_target, shape_emb_target, appearance_emb_target, abs_pose_target = batch
    seg_output, depth_output, small_disp_output, heatmap_output, shape_emb_output, appearance_emb_output, abs_pose_output = self.forward(image)

    log = {}
    logger = self.logger.experiment
    with torch.no_grad():
      prefix_loss = 'validation'

      loss = compute_depth_loss(self.disp_loss, depth_output, copy.deepcopy(depth_target), log, f'{prefix_loss}_detailed/loss/train_cost_volume_disp', self.hyperparams)
      # if self.hyperparams.depth_output is None:
      #   loss = loss + small_depth_output.compute_loss(depth_target, log, f'{prefix}_detailed/loss/train_cost_volume_disp')

      if self.hyperparams.frozen_stereo_checkpoint is None:
        loss = loss + compute_depth_loss(self.disp_loss, small_disp_output, depth_target, log, f'{prefix_loss}_detailed/loss/train_cost_volume_disp', self.hyperparams)

      loss = loss + compute_seg_loss(seg_output, seg_target, log, f'{prefix_loss}_detailed/loss/seg', self.hyperparams)
      
      loss = loss + compute_pose_shape_loss(self, heatmap_target, shape_emb_target, appearance_emb_target, abs_pose_target, heatmap_output, shape_emb_output, appearance_emb_output, abs_pose_output, log, f'{prefix_loss}_detailed/pose', self.hyperparams)
      
      # loss = depth_output.compute_loss(copy.deepcopy(depth_target), log, f'{prefix_loss}_detailed/loss/refined_disp')
      # if self.hyperparams.frozen_stereo_checkpoint is None:
      #   loss = loss + small_depth_output.compute_loss(depth_target, log, f'{prefix_loss}_detailed_loss/train_cost_volume_disp')
      # loss = loss + seg_output.compute_loss(seg_target, log, f'{prefix_loss}_detailed/loss/seg')      
      # if pose_targets[0] is not None:
      #   loss = loss + pose_outputs.compute_loss(pose_targets, log, f'{prefix_loss}_detailed/pose')

      log['validation/loss/total'] = loss.item()

      rank = dist.get_rank()
      if rank == 0 and batch_idx <5:
        llog = {}
        prefix = f'val/{batch_idx}'
        left_image_np = extract_left_numpy_img(image[0])
        
        seg_pred_vis = get_seg_visualization_img(left_image_np, seg_output)
        llog[f'{prefix}/seg'] = wandb.Image(seg_pred_vis, caption=prefix)

        depth_vis = get_depth_visualization_img(left_image_np, depth_output)
        llog[f'{prefix}/disparity'] = wandb.Image(depth_vis, caption=prefix)

        small_depth_vis = get_depth_visualization_img(left_image_np, small_disp_output)
        llog[f'{prefix}/small_disparity'] = wandb.Image(small_depth_vis, caption=prefix)
        logger.log(llog)
        logger.log(log)

      #TODO fix visualization
      # if batch_idx < 5:
      #   llog = {}
      #   left_image_np = extract_left_numpy_img(image[0])
      #   prefix = f'val/{batch_idx}'
      #   depth_vis = depth_output.get_visualization_img(np.copy(left_image_np))
      #   llog[f'{prefix}/disparity'] = wandb.Image(depth_vis, caption=prefix)
      #   small_depth_vis = small_depth_output.get_visualization_img(np.copy(left_image_np))
      #   llog[f'{prefix}/small_disparity'] = wandb.Image(small_depth_vis, caption=prefix)
      #   self.eval_metrics.draw_detections(
      #      seg_output,left_image_np, llog, prefix
      #   )
      #   logger.log(llog)
    return log

  def validation_epoch_end(self, outputs):
    self.trainer.checkpoint_callback.save_best_only = False
    mean_dict = {}
    for key in outputs[0].keys():
        mean_dict[key] = np.mean([d[key] for d in outputs], axis=0)
    logger = self.logger.experiment
    logger.log(mean_dict)
    log = {}
    return {'log': log}

  def train_dataloader(self):
    return common.get_loader(
        self.hyperparams,
        "train",
        preprocess_func=self.preprocess_func,
        datapoint_dataset=self.train_dataset
    )


  def val_dataloader(self):
    return common.get_loader(self.hyperparams, "val", preprocess_func=self.preprocess_func)

  # def configure_optimizers(self):
  #   optimizer = torch.optim.Adam(self.parameters(), lr=self.hyperparams.optim_learning_rate)
  #   lr_lambda = lambda_learning_rate_poly(self.epochs, self.hyperparams.optim_poly_exp)
  #   if self.hyperparams.optim_warmup_epochs is not None and self.hyperparams.optim_warmup_epochs > 0:
  #     lr_lambda = lambda_warmup(self.hyperparams.optim_warmup_epochs, 0.2, lr_lambda)
  #   scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
  #   return [optimizer], [scheduler]

  def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=self.hyperparams.optim_learning_rate)
      lr_lambda = lambda_learning_rate_poly(self.epochs, self.hyperparams.optim_poly_exp)
      if self.hyperparams.optim_warmup_epochs is not None and self.hyperparams.optim_warmup_epochs > 0:
          lr_lambda = lambda_warmup(self.hyperparams.optim_warmup_epochs, 0.2, lr_lambda)
      self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
      return optimizer
