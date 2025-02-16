'''
Author: DyllanElliia
Date: 2024-07-05 16:30:37
LastEditors: DyllanElliia
LastEditTime: 2024-07-05 16:35:46
Description: 
'''
"""
Author: DyllanElliia
Date: 2023-11-28 20:33:59
LastEditors: DyllanElliia
LastEditTime: 2023-11-28 20:33:59
Description: 
"""
"""
Author: DyllanElliia
Date: 2023-03-18 15:44:56
LastEditors: DyllanElliia
LastEditTime: 2023-07-31 14:23:31
Description: 
"""
import os
import time
import argparse
from utils.misc import *
from utils.evaluate import *

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--save_title", type=str_list, default=["/"])
parser.add_argument("--output_root",
                    type=str,
                    default="./outres/results/others")
parser.add_argument("--dataset_root", type=str, default="./datas")
parser.add_argument("--dataset", type=str, default="PUNet")
parser.add_argument("--resolution", type=str, default="10000_poisson")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()
# python eval.py --save_title MODNet_50k_0.01 --dataset MODNet --resolution 50k --output_root ./outres/results/ScoreBased

evaluator = Evaluator(
    dataset_root=args.dataset_root,
    dataset=args.dataset,
    summary_dir=args.output_root,
    device=args.device,
    res_gts=args.resolution,
)

print(args.save_title)

for dir in args.save_title:
  output_dir = os.path.join(args.output_root, dir)
  logger = get_logger("test", output_dir)
  evaluator.logger = logger
  evaluator.reLoad_output(os.path.join(output_dir, "pcl"), dir)
  evaluator.run()
