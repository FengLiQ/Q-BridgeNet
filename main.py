from config import get_args
from segtok import segtok
from finetuning import finetuning
from evaluation import evaluation

from datasets import S2T_Dataset
import torch
from showing import showimg
from segtok import segment
import time

if __name__ == '__main__':
    args = get_args()
    if args.function == "segtok":
        segtok(args)
    elif args.function == "finetune":
        finetuning(args)
    elif args.function == "evaluate":
        evaluation(args)

    '''
    dgs_data = S2T_Dataset(label_path="data/Phoenix-2014T/labels.train", pose_dirs="Phoenix-2014T")
    pose = dgs_data[18][1]
    showimg(pose, dgs_data[18][0] + str(time.time()))
    tok_model = torch.load("__DGS__token.pt")
    segment(dgs_data, tok_model, "t.pkl")
    '''
