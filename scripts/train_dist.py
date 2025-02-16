import os
import sys
import time

import numpy as np
import torch.backends
import torch.backends
from torch.optim.optimizer import Optimizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
import torch.distributed as dist
import torch.nn as nn

from models.sbdnet import SBD, releaseGpuCache
from models.schedule import VarianceSchedule
from datasets import *
from utils.misc import *
from utils.transforms import *

# from utilsbd.denoise import *
from models.utils import chamfer_distance_unit_sphere

# gpus = [2, 4]
# torch.cuda.set_device("cuda:{}".format(gpus[0]))

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, default="SBD", help="training run name")
# Dataset
parser.add_argument("--dataset_root", type=str, default="../datas")
parser.add_argument("--dataset", type=str, default="PUNet")
parser.add_argument("--patch_size", type=int, default=1000)
parser.add_argument(
    "--resolutions",
    type=str_list,
    default=["10000_poisson", "30000_poisson", "50000_poisson"],
)
parser.add_argument("--noise_min", type=float, default=0.0)
parser.add_argument("--noise_max", type=float, default=0.0)
parser.add_argument("--train_batch_size", type=int, default=48)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--aug_rotate",
                    type=eval,
                    default=True,
                    choices=[True, False])
parser.add_argument("--patch_ratio", type=float, default=1.2)
# Model architecture
## Score-based
# parser.add_argument('--supervised', type=eval, default=True, choices=[True, False])
parser.add_argument("--frame_knn", type=int, default=32)
parser.add_argument("--sample_knn", type=int, default=32)
parser.add_argument("--edge_conv_knn", type=int, default=16)
parser.add_argument("--conv_num_fc_layers", type=int, default=3)
parser.add_argument("--conv_growth_rate", type=int, default=12)
parser.add_argument("--num_convs", type=int, default=4)
parser.add_argument("--conv_channels", type=int, default=24)
parser.add_argument("--mlp_feat",
                    type=eval,
                    default=False,
                    choices=[True, False])
parser.add_argument("--cat_feat",
                    type=eval,
                    default=False,
                    choices=[True, False])
parser.add_argument("--num_train_points", type=int, default=128)
parser.add_argument("--num_clean_nbs",
                    type=int,
                    default=4,
                    help="For supervised training.")
# parser.add_argument('--num_selfsup_nbs', type=int, default=8, help='For self-supervised training.')
parser.add_argument("--dsm_sigma", type=float, default=0.01)
parser.add_argument("--dsm_sigma2", type=float, default=0.01)
parser.add_argument("--displacement_alpha", type=float, default=0.01)
parser.add_argument("--score_net_hidden_dim", type=int, default=128)
parser.add_argument("--score_net_hidden_h", type=int, default=128)
parser.add_argument("--score_net_decoder_h", type=int, default=0)
parser.add_argument("--score_net_num_blocks", type=int, default=4)
parser.add_argument("--normalize_patch", type=bool, default=True)
## Diffusion
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--beta_1", type=float, default=5e-8)
parser.add_argument("--beta_T", type=float, default=2.4e-6)
parser.add_argument("--sched_mode", type=str, default="linear")
parser.add_argument("--flexibility", type=float, default=0.0)
parser.add_argument("--residual",
                    type=eval,
                    default=True,
                    choices=[True, False])
# parser.add_argument(
#     '--compute_feat_eachStep', type=eval, default=False, choices=[True, False]
# )
# parser.add_argument(
#     '--loss_use_displacement', type=eval, default=False, choices=[True, False]
# )
parser.add_argument("--use_sigmaLoss",
                    type=eval,
                    default=False,
                    choices=[True, False])
# parser.add_argument('--use_1dGausssE', type=eval, default=False, choices=[True, False])
parser.add_argument("--use_patch_ratio",
                    type=eval,
                    default=True,
                    choices=[True, False])
parser.add_argument("--use_eqdis",
                    type=eval,
                    default=False,
                    choices=[True, False])
parser.add_argument("--alpha", type=float, default=0.99)
parser.add_argument("--Tgen_mode", type=str, default="common")
# Optimizer and scheduler

parser.add_argument("--sched_type", default="const", type=str)
parser.add_argument("--sched_patience", default=5, type=int)
parser.add_argument("--sched_factor", default=0.5, type=float)
parser.add_argument("--opt_type", type=str, default="adam")
parser.add_argument("--min_lr", default=1e-7, type=float)
parser.add_argument("--threshold", default=1e-5, type=float)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--max_grad_norm", type=float, default=100)
# parser.add_argument('--end_lr', type=float, default=1e-4)
# parser.add_argument('--sched_start_epoch', type=int, default=150 * THOUSAND)
# parser.add_argument('--sched_end_epoch', type=int, default=300 * THOUSAND)
# Training
parser.add_argument("--seed", type=int, default=2023)
parser.add_argument("--logging", type=eval, default=True, choices=[True, False])
parser.add_argument("--log_root", type=str, default="./logs")
# parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--local_rank",
                    default=-1,
                    type=int,
                    help="node rank for distributed training")
# parser.add_argument("--mulDevice", type=str, default=None)
parser.add_argument("--max_iters", type=int, default=1.5 * MILLION)
parser.add_argument("--val_freq", type=int, default=100000)

# parser.add_argument('--val_upsample_rate', type=int, default=4)
parser.add_argument("--val_num_visualize", type=int, default=8)
parser.add_argument("--val_noise", type=float, default=0.015)
# parser.add_argument('--ld_step_size', type=float, default=0.2)
parser.add_argument("--tag", type=str, default=None)
parser.add_argument("--note", type=str, default="train")
parser.add_argument("--resume", type=eval, default=False, choices=[True, False])
parser.add_argument("--ckpt", type=str, default="")
parser.add_argument("--use_static_featNet",
                    type=eval,
                    default=False,
                    choices=[True, False])
parser.add_argument("--feat_ckpt", type=str, default="")
parser.add_argument("--use_patch", type=bool, default=False)
args = parser.parse_args()

# local_rank=args.local_rand
rank = 0
local_rank = 0
world_size = 1


def init_distributed_mode(args):
  # 如果是多机多卡的机器，WORLD_SIZE代表使用的机器数，RANK对应第几台机器
  # 如果是单机多卡的机器，WORLD_SIZE代表有几块GPU，RANK和LOCAL_RANK代表第几块GPU
  if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    # LOCAL_RANK代表某个机器上第几块GPU
    local_rank = int(os.environ['LOCAL_RANK'])
  elif 'SLURM_PROCID' in os.environ:
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = rank % torch.cuda.device_count()
  else:
    print('Not using distributed mode')
    args.distributed = False
    return

  args.distributed = True

  torch.cuda.set_device(local_rank)  # 对当前进程指定使用的GPU
  args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
  dist.init_process_group(backend="nccl")
  dist.barrier()  # 等待每个GPU都运行完这个地方以后再继续
  return rank, local_rank, world_size


rank, local_rank, world_size = init_distributed_mode(args=args)
print(rank, local_rank, world_size)
seed_all(args.seed + local_rank)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dist.barrier()

# dist.init_process_group(backend="nccl")
# torch.cuda.set_device(local_rank)
args.lr *= world_size  # 学习率要根据并行GPU的数倍增

if local_rank == 0:
  # Logging
  if args.logging:
    log_dir = get_new_log_dir(
        args.log_root,
        prefix="D%s_" % (args.dataset),
        postfix="_" + args.tag if args.tag is not None else "",
    )
    logger = get_logger("train", log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, log_dir, args)
  else:
    logger = get_logger("train", None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
  logger.info(args)
  logger.info("note: " + args.note)

  # Model
  logger.info("Building model...")
# if args.mulDevice:
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.mulDevice
#     if torch.cuda.device_count() > 1:
#         device = args.device
#         print("Let's use", torch.cuda.device_count(), "GPUs!")
#         net = torch.nn.DataParallel(SBD(args)).to(device)
#         if isinstance(net, torch.nn.DataParallel):
#             model = net.module
# else:
#     device = args.device
#     model = SBD(args).to(device)
device = f"cuda:{local_rank}"
model = SBD(args).cuda()
# model.dsb.scoreNet = torch.nn.parallel.DistributedDataParallel(
#     model.dsb.scoreNet, device_ids=[local_rank]
# )
# model.dsb.featureNet = torch.nn.parallel.DistributedDataParallel(
#     model.dsb.featureNet, device_ids=[local_rank]
# )
if local_rank == 0:
  logger.info(repr(model))

if world_size > 1:
  checkpoint_path = "./initial_weights.pt"
  # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
  if rank == 0:
    torch.save(model.state_dict(), checkpoint_path)

  dist.barrier()
  # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
  model.load_state_dict(
      torch.load(checkpoint_path, map_location=torch.device(device)))

if args.resume:
  if local_rank == 0:
    logger.info("ft ckpt:{}".format(args.ckpt))
  ckpt = torch.load(args.ckpt, map_location=torch.device(device))
  # if args.mulDevice:
  #     net.load_state_dict(ckpt["state_dict"])
  # else:
  model.load_state_dict(ckpt["state_dict"])

# if args.feat_ckpt != "":
#   if local_rank == 0:
#     logger.info("Loading featureNet...")
#   feat_ckpt = torch.load(args.feat_ckpt, map_location=torch.device(device))
#   model.dsb.featureNet.load_state_dict(feat_ckpt)
#   if args.use_static_featNet:
#     if local_rank == 0:
#       logger.info("Freeze featureNet...")
#     model.dsb.featureNet._requires_grad_ = False
#     model.dsb.featureNet.requires_grad_(False)

if args.feat_ckpt != "":
  if local_rank == 0:
    logger.info("Loading featureNet...")
  feat_ckpt = torch.load(args.feat_ckpt, map_location=torch.device(device))
  model.dsb.load_feat_net(feat_ckpt)
  if args.use_static_featNet:
    if local_rank == 0:
      logger.info("Freeze featureNet...")
    # model.dsb.featureNet._requires_grad_ = False
    # model.dsb.featureNet.requires_grad_(False)
    model.dsb.set_feat_net_requires_grad(False)

model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=[local_rank])
# if torch.cuda.is_available():
#     torch.backends.cudnn.benchmark = True

# Datasets and loaders
# NOTE: Noise Point Cloud only for validate.
if local_rank == 0:
  logger.info("Loading datasets")
train_dset = PairedPatchDataset(
    datasets=[
        PointCloudDataset(
            root=args.dataset_root,
            dataset=args.dataset,
            split="train",
            resolution=resl,
            transform=standard_train_transforms(
                noise_std_max=args.noise_min,
                noise_std_min=args.noise_min,
                rotate=args.aug_rotate,
            ),
        ) for resl in args.resolutions
    ],
    patch_size=args.patch_size,
    patch_ratio=args.patch_ratio,
    on_the_fly=True,
)
if local_rank == 0:
  val_dset = PointCloudDataset(
      root=args.dataset_root,
      dataset=args.dataset,
      split="test",
      resolution=args.resolutions[-1],
      transform=standard_train_transforms(
          noise_std_max=args.val_noise,
          noise_std_min=args.val_noise,
          rotate=False,
          scale_d=0,
      ),
  )
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dset,
                                                                shuffle=True)
train_iter = get_data_iterator(
    DataLoader(
        train_dset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        # shuffle=True,
        sampler=train_sampler,
    ))

# Optimizer and scheduler
# optimizer = torch.optim.Adam(
#     model.parameters(),
#     lr=args.lr,
#     weight_decay=args.weight_decay,
# )
if args.opt_type == "AdamW":
  if local_rank == 0:
    logger.info("use AdamW")
  optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=args.lr,
      weight_decay=args.weight_decay,
  )
elif args.opt_type == "AdamW2":
  params = [{
      "params": model.module.dsb.scoreNet.parameters(),
      "lr": args.lr
  }, {
      "params": model.module.dsb.featureNet.parameters(),
      "lr": args.lr / 5.0
  }, {
      "params": model.module.dsb.fusion.parameters(),
      "lr": args.lr
  }]
  optimizer = torch.optim.AdamW(
      params,
      lr=args.lr,
      weight_decay=args.weight_decay,
  )
elif args.opt_type == "SGD":
  if local_rank == 0:
    logger.info("use SGD")
  optimizer = torch.optim.SGD(
      model.parameters(),
      lr=args.lr,
      weight_decay=args.weight_decay,
  )
else:
  if local_rank == 0:
    logger.info("use Adam")
  optimizer = torch.optim.Adam(
      model.parameters(),
      lr=args.lr,
      weight_decay=args.weight_decay,
  )
if args.sched_type == "cosine":
  if local_rank == 0:
    logger.info("use cosine LR")
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=args.val_freq *
                                                         args.sched_patience,
                                                         eta_min=1e-7)
elif args.sched_type == "linear":
  if local_rank == 0:
    logger.info("use linear LR")

  # class linearLR:

  #   def __init__(self,
  #                optimizer,
  #                begin_lr,
  #                end_lr,
  #                last_epoch: int = ...) -> None:
  #     self.opt = optimizer
  #     self.current_epoch = 0
  #     self.last_epoch = last_epoch
  #     self.begin_lr = begin_lr
  #     self.end_lr = end_lr

  #   def step(self):
  #     self.current_epoch += 1
  #     it = min(math.fmod(self.current_epoch / self.last_epoch, 1.0), 1.0)
  #     for param_group in self.opt.param_groups:
  #       param_group['lr'] = self.begin_lr * (1.0 - it) + it * self.end_lr

  # scheduler = linearLR(optimizer, args.lr, 1e-7,
  #                      args.val_freq * args.sched_patience)
  def lr_linear(epoth):
    global args
    i = epoth / args.max_iters
    return max(1 - i, 1e-7)

  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                lr_lambda=lr_linear,
                                                last_epoch=-1)
elif args.sched_type == "anneal":
  if local_rank == 0:
    logger.info("use anneal LR")

  def lr_cos(epoth):
    global args
    i = epoth / args.max_iters
    ci = (math.cos(i * math.pi) + 1) / 2
    return max(ci, 1e-2)

  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                lr_lambda=lr_cos,
                                                last_epoch=-1)
else:
  if local_rank == 0:
    logger.info("use const LR")
  scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer,
                                                  factor=1.0,
                                                  total_iters=args.max_iters)
lossLog = []
cdLog = []


def reduce_tensor(tensor: torch.Tensor):
  rt = tensor.clone()
  torch.utils.data.distributed.all_reduce(
      rt, op=torch.utils.data.distributed.reduce_op.SUM)
  rt /= torch.utils.data.distributed.get_world_size()
  return rt


def reduce_value(value, average=True):
  # world_size = torch.utils.data.distributed.get_world_size()
  if world_size < 2:  # 单GPU的情况
    return value

  with torch.no_grad():
    dist.all_reduce(value)  # 对不同设备之间的value求和
    if average:  # 如果需要求平均，获得多块GPU计算loss的均值
      value /= world_size

    return value


# Train, validate and test
def train(it):
  # loss = 0
  # Load data
  batch = next(train_iter)
  # pcl_noisy = batch['pcl_noisy'].to(device)
  pcl_clean = batch["pcl_clean"].cuda(non_blocking=True)
  # Reset grad and model state
  optimizer.zero_grad(set_to_none=True)
  model.train()

  # Forward
  loss = model.module.get_loss(pcl_clean)
  # reduced_loss = reduce_tensor(loss_i.data)
  # loss += reduced_loss.item()
  # loss = loss_i.mean()

  # Backward and optimize
  loss.backward()
  loss = reduce_value(loss, average=True)
  orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
  optimizer.step()
  scheduler.step()
  # if it % 1000 == 0:
  #     scheduler.step(loss)
  if local_rank == 0:
    # Logging
    logger.info("[Train] Iter %04d | Loss %.4f | Grad %.4f | lr %.6f" % (
        it,
        loss.item(),
        orig_grad_norm,
        optimizer.param_groups[0]["lr"],
    ))
    # lossLog.append(loss.item())
    writer.add_scalar("train/loss", loss, it)
    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], it)
    writer.add_scalar("train/grad_norm", orig_grad_norm, it)
    writer.flush()


def grad_compute(grad_pred, x_t, x_m_a, Patchs_idx, k):
  """
    Args:
        grad_pred:  Input grad prediction, (B, N, K, 3)
        x_t: Point clouds, (B, N, 3)
        Patchs_idx: Point Patch indices, (B, N, K)
    """
  # return grad_pred.mean(2).squeeze(2)
  grad = grad_pred.contiguous().view(x_t.size(0), -1,
                                     x_t.size(2))  # (B, N*K, 3)
  nn_idx = Patchs_idx.view(x_t.size(0), -1)  # (B, N*K)
  nn_idx = nn_idx.unsqueeze(-1).expand_as(grad)  # (B, N*K, 1) -> (B, N*K, 3)

  num_cnt = torch.ones_like(grad)
  num_grads = torch.zeros_like(x_t)
  num_grads.scatter_add_(dim=1, index=nn_idx, src=num_cnt)
  num_grads[num_grads < 1] = 1
  acc_grads = torch.zeros_like(x_t)
  acc_grads.scatter_add_(dim=1, index=nn_idx, src=grad)
  acc_grads = acc_grads / num_grads
  # print(grad.min(), grad.max())
  # print(acc_grads.min(), acc_grads.max())
  return acc_grads


use_patch = args.use_patch

# Main loop
if local_rank == 0:
  logger.info("Start training...")
try:
  torch.cuda.synchronize(device)
  dist.barrier()
  for it in range(1, args.max_iters + 1):
    # print(local_rank, it)
    train(it)
    # if it % 10000 == 0:
    # scheduler.step()

    if it % val_freq == 0 or it == args.max_iters:
      torch.cuda.synchronize(device)
      dist.barrier()
      # cd_loss_f = validate_fast(it)
      # # cd_loss = validate(it)
      # cd_loss = validate_iter(it)
      # # [patch, all]
      # cd_loss.append(cd_loss_f[0])  # [iter,fast]
      if local_rank == 0:
        cd_loss = validate_iter_tf(it,
                                   val_dset=val_dset,
                                   val_noise=1,
                                   use_patch_base=use_patch)[0],

        print("local rank:", local_rank)
        # [patch, all]
        opt_states = {
            "optimizer": optimizer.state_dict(),
            # "scheduler": scheduler.state_dict(),
        }
        # print("local rank:", local_rank)
        # if torch.distributed.get_rank() == 0:
        # if
        if local_rank != 0:
          continue
        if cd_loss[0] > best0 and cd_loss[1] > best1 and args.resume == False:
          continue
        best0 = np.min([cd_loss[0], best0])
        best1 = np.min([cd_loss[1], best1])
        model_to_save = model.module if hasattr(model, "module") else model
        ckpt_mgr.save(
            model_to_save,
            args,
            int(cd_loss[0] * 1e7) + cd_loss[1] * 10,
            opt_states,
            step=it,
        )
        torch.save({
            'args': args,
            'state_dict': model.state_dict(),
        }, os.path.join(ckpt_mgr.save_dir, "ckpt_last.pt"))

      torch.cuda.synchronize(device)
      dist.barrier()
      releaseGpuCache()
      releaseGpuCache()
      releaseGpuCache()
      torch.cuda.synchronize(device)
      dist.barrier()
      # if best1 < 0.000077 or best0 < 0.000074:
      #   val_freq = args.val_freq / 2
      # if best1 > 0.0001 or best0 > 0.0001:
      #   optimizer.param_groups[0]["lr"] = 1e-4
      # elif best1 > 0.00009 or best0 > 0.00009:
      #   optimizer.param_groups[0]["lr"] = 5e-5
      # elif best1 > 0.000085 or best0 > 0.000085:
      #   optimizer.param_groups[0]["lr"] = 1e-5
      # elif best1 > 0.000083 or best0 > 0.000083:
      #   optimizer.param_groups[0]["lr"] = 5e-6
      # elif best1 > 0.000080 or best0 > 0.000080:
      #   optimizer.param_groups[0]["lr"] = 1e-6
      # elif best1 > 0.000070 or best0 > 0.000070:
      #   optimizer.param_groups[0]["lr"] = 5e-7
      # else:
      #   optimizer.param_groups[0]["lr"] = 1e-3
      # if best0 < 0.000073 and best1 < 0.000077:
      #   optimizer.param_groups[0]["lr"] = 1e-6
      #   break
      # if it >= 1000000:
      #     val_freq/=5
      # ckpt_mgr.save(model, args, 0, opt_states, step=it)
      # draw1Dline(lossLog, 5, "./outres/loss")
      # draw1Dline(cdLog, 4, "./outres/cd")

except KeyboardInterrupt:
  # draw1Dline(lossLog, 5, "./outres/loss", y_max=0.2)
  # draw1Dline(cdLog, 4, "./outres/cd")
  if local_rank == 0:
    logger.info("Terminating...")
