import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from models.sbdnet import SBD
from models.sbdiffusion import VarianceSchedule
from datasets import *
from utils.misc import *
from utils.transforms import *
from utils.evaluate import *
import pytorch3d.ops

# from utilsbd.denoise import *
from models.utils import chamfer_distance_unit_sphere


def input_iter(input_dir):
  for fn in os.listdir(input_dir):
    if fn[-3:] != "xyz":
      continue
    pcl_noisy = torch.FloatTensor(np.loadtxt(os.path.join(input_dir, fn)))
    pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_noisy)
    yield {
        "pcl_noisy": pcl_noisy,
        "name": fn[:-4],
        "center": center,
        "scale": scale,
    }


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="../pretrained/ckpt.pt")
parser.add_argument(
    "--input_root",
    type=str,
    default="../datas/examples",
)
parser.add_argument("--output_root",
                    type=str,
                    default="../outres/results/others")
parser.add_argument("--dataset_root", type=str, default="../datas")
parser.add_argument("--dataset", type=str, default="PUNet")
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--resolution", type=str, default="10000_poisson")
parser.add_argument("--noise", type=float, default=0.02)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=2023)
# Denoiser parameters
parser.add_argument("--seed_k", type=int, default=3)
parser.add_argument("--patch_size", type=int, default=1000)
parser.add_argument("--niters", type=int, default=1)
parser.add_argument(
    "--sample_knn",
    type=int,
    default=4,
    help="Number of score functions to be ensembled",
)
parser.add_argument("--dsm_sigma", type=float, default=0.01)
parser.add_argument("--num_steps", type=int, default=20)
parser.add_argument("--beta_1", type=float, default=1e-7)
parser.add_argument("--beta_T", type=float, default=0)
parser.add_argument("--kappa", type=float, default=2)
parser.add_argument("--round", type=float, default=5)
parser.add_argument("--sched_mode", type=str, default="linear")
parser.add_argument("--flexibility", type=float, default=0.0)
parser.add_argument("--feature_fusion",
                    type=eval,
                    default=True,
                    choices=[True, False])
# parser.add_argument("--is_sigma_eta_zero",
#                     type=eval,
#                     default=False,
#                     choices=[True, False])
parser.add_argument("--eta", type=float, default=0)
# parser.add_argument("--tag", type=str, default="")
args = parser.parse_args()
seed_all(args.seed)

# Input/Output
input_dir = os.path.join(
    args.input_root, "%s_%s_%s" % (args.dataset, args.resolution, args.noise))
# input_dir = os.path.join(args.input_root, '%s_%s' % (args.dataset, args.resolution))

save_title = "{dataset}_{res}_{noise}_{time}".format_map({
    "dataset": args.dataset,
    "res": args.resolution,
    "noise": args.noise,
    "time": time.strftime("%m-%d-%H-%M-%S", time.localtime()),
})
output_dir = os.path.join(args.output_root, save_title)
os.makedirs(output_dir)
os.makedirs(os.path.join(output_dir, "pcl"))  # Output point clouds
logger = get_logger("test", output_dir)
for k, v in vars(args).items():
  logger.info("[ARGS::%s] %s" % (k, repr(v)))

# Model
ckpt = torch.load(args.ckpt, map_location=args.device)
model = SBD(ckpt["args"]).to(args.device)
model.load_state_dict(ckpt["state_dict"])

if False:
  print("export featureNet checkpoint")
  torch.save(model.dsb.featureNet.state_dict(), "../pretrained/featureNet_.pt")

if args.tag != "":
  model.dsb.tag = args.tag
  print("tag:", model.dsb.tag)

# Denoise
# ld_step_size = (
#     args.ld_step_size if args.ld_step_size is not None else ckpt['args'].ld_step_size
# )
# logger.info('ld_step_size = %.8f' % ld_step_size)
sample_knn = args.sample_knn
# args.noise=args.noise

timeacc = 0
timecnt = 0
import time

cnt = 0

for data in input_iter(input_dir):
  logger.info("[%d] name:%s" % (cnt, data["name"]))
  cnt += 1
  pcl_noisy = data["pcl_noisy"].to(args.device)
  # if data['name'] != 'angel2_aligned' and data['name'] != 'david':
  #     continue
  # if data['name'] != 'david':
  #     continue

  with torch.no_grad():
    model.eval()
    pcl_next = pcl_noisy

    # v1
    tb = time.time()
    if args.beta_T == 0.0:
      sigma_T = model.estimate_sigma(pcl_next, sample_knn, kappa=args.kappa)
    else:
      print("Please check this part")
      sigma_T = 0.0
    logger.info("sigma:%f" % (sigma_T))

    # model.change_vs(args, sigma_T / args.niters)
    # for _ in range(args.niters):
    #     pcl_next = model.sample(
    #         pcl_next,
    #         ret_traj=False,
    #         sample_knn=sample_knn,
    #         kappa=args.kappa,
    #         is_sigma_eta_zero=args.is_sigma_eta_zero,
    #         round=1,
    #         use_nearest_feat=False,
    #     )[0]
    model.change_vs(args, sigma_T)
    logger.info("sigma_T:%f" % (model.dsb.var_sched.betas[-1]))
    pcl_next = model.sample(
        pcl_next,
        ret_traj=False,
        sample_knn=sample_knn,
        kappa=args.kappa,
        is_sigma_eta_zero=True if args.eta == 0 else False,
        eta=args.eta,
        round=1,
        feature_fusion=args.feature_fusion,
        use_nearest_feat=False,
        use_patch_base=False
        if args.dataset not in ["Kinectv1", "Kinectv2"] else True,
    )[0]

    pcl_denoised = pcl_next.cpu()
    # Denormalize
    pcl_denoised = pcl_denoised * data["scale"] + data["center"]
    te = time.time()
    timeacc += te - tb
    timecnt += 1
  save_path = os.path.join(output_dir, "pcl", data["name"] + ".xyz")
  np.savetxt(save_path, pcl_denoised.numpy(), fmt="%.8f")
  # save_path = os.path.join(output_dir, 'pcl', data['name'] + '_noise.xyz')
  # pcl_noisy = pcl_noisy.cpu() * data['scale'] + data['center']
  # np.savetxt(save_path, pcl_noisy.numpy(), fmt='%.8f')
  # save_path = os.path.join(output_dir, 'pcl', data['name'] + '_clear.xyz')
  # np.savetxt(save_path, data['pcl_clear'].numpy(), fmt='%.8f')
print(timeacc / timecnt)
# Evaluate
evaluator = Evaluator(
    output_pcl_dir=os.path.join(output_dir, "pcl"),
    dataset_root=args.dataset_root,
    dataset=args.dataset,
    summary_dir=args.output_root,
    experiment_name=save_title,
    device=args.device,
    res_gts=args.resolution,
    logger=logger,
)
evaluator.run()
