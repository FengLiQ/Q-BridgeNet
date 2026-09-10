import argparse
import torch
import os
import torch
import torch.nn as nn
import torch.distributed as dist

DEVICE = torch.device(f"cuda")
if "WORLD_SIZE" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

dist.init_process_group(backend="nccl")
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(LOCAL_RANK)
DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
RANK = dist.get_rank()
WORLD_SIZE = dist.get_world_size()
print("LOCAL_RANK:", LOCAL_RANK)
print("RANK:", RANK)
print("WORLD_SIZE:", WORLD_SIZE)
print("DEVICE:", DEVICE)

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

POSE_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
FACE_LIST = [23, 26, 29, 33, 36, 39, 41, 43, 46, 48, 56, 59, 62, 65, 68, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 53]
LEFT_LIST = [91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
RIGHT_LIST = [112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]
POSE_POINT_SIZE = len(POSE_LIST)  # 11 for pose
FACE_POINT_SIZE = len(FACE_LIST)  # 26 for face
HAND_POINT_SIZE = len(LEFT_LIST)  # 21 for each hand
FEATURE_POINT_SIZE = POSE_POINT_SIZE + FACE_POINT_SIZE + HAND_POINT_SIZE + HAND_POINT_SIZE

LANG_LABEL = ["__ASL__", "__CSL__", "__DGS__"]
MAX_CLIP_LEN = 32
MAX_SIGN_LEN = 256


def get_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--function", type=str,default=None, help="module name")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--asl_label", type=str, default=None, help="asl label")
    parser.add_argument("--asl_dir", type=str, default=None, help="asl data dir")
    parser.add_argument("--asl_seg", type=str, default=None, help="asl segmentation label")
    parser.add_argument("--asl_tok", type=str, default=None, help="asl tokenization model")
    parser.add_argument("--asl_ex_label", type=str, default=None, help="asl extended translation")
    parser.add_argument("--csl_label", type=str, default=None, help="csl label")
    parser.add_argument("--csl_dir", type=str, default=None, help="csl data dir")
    parser.add_argument("--csl_seg", type=str, default=None, help="csl segmentation label")
    parser.add_argument("--csl_tok", type=str, default=None, help="csl tokenization model")
    parser.add_argument("--csl_ex_label", type=str, default=None, help="csl extended translation")
    parser.add_argument("--dgs_label", type=str, default=None, help="dgs label")
    parser.add_argument("--dgs_dir", type=str, default=None, help="dgs data dir")
    parser.add_argument("--dgs_seg", type=str, default=None, help="dgs segmentation label")
    parser.add_argument("--dgs_tok", type=str, default=None, help="dgs tokenization model")
    parser.add_argument("--dgs_ex_label", type=str, default=None, help="dgs extended translation")
    parser.add_argument("--output", type=str, default="output", help="output dir")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B", help="llm name")
    parser.add_argument("--resume", type=str, default=None, help="resume training")

    return parser.parse_args()
