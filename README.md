# Q-BridgeNet



## Pipeline



1. segtok: train RVQ tokenizers and segment pose sequences  

2. finetune: inject quantized pose features into the LLM and LoRA-finetune  



Entry point: `main.py`.



## Data



Download the How2Sign, CSL-Daily, and Phoenix14T datasets based on your requirements.

The folders for the How2Sign, CSL-Daily datasets can be downloaded from https://huggingface.co/ZechengLi19/Uni-Sign, which is used as one of our baselines.

For Phoenix14T, it can be downloaded from https://drive.google.com/file/d/1zzkvf-j-oAubY_qid2tIYzU5YNLlKBHu

We provide the GPT-5-generated multilingual labels (*_ex_labels.pkl).


## Usage

### for segmentation and tokenization (ini)

python main.py --function segtok --asl_label data/How2Sign/labels.train --asl_dir How2Sign --csl_label data/CSL_Daily/labels.train --csl_dir CSL_Daily --dgs_label data/Phoenix-2014T/labels.train --dgs_dir Phoenix-2014T --epochs 5

### for segmentation and tokenization (resume) (if --epochs 0 for given segmentation, then tokenization only)
python main.py --function segtok --asl_label data/How2Sign/labels.train --asl_dir How2Sign --asl_seg __ASL__seg.pkl --csl_label data/CSL_Daily/labels.train --csl_dir CSL_Daily --csl_seg __CSL__seg.pkl --dgs_label data/Phoenix-2014T/labels.train --dgs_dir Phoenix-2014T --dgs_seg __DGS__seg.pkl --epochs 5


### for finetuning
torchrun --nproc_per_node=4 main.py --function finetune --asl_label data/How2Sign/labels.train --asl_dir How2Sign --asl_seg __ASL__seg.pkl --asl_tok __ASL__token.pt --asl_ex_label data/How2Sign/ASL_ex_labels.pkl --csl_label data/CSL_Daily/labels.train --csl_dir CSL_Daily --csl_seg __CSL__seg.pkl --csl_tok __CSL__token.pt --csl_ex_label data/CSL_Daily/CSL_ex_labels.pkl --dgs_label data/Phoenix-2014T/labels.train --dgs_dir Phoenix-2014T --dgs_seg __DGS__seg.pkl --dgs_tok __DGS__token.pt --dgs_ex_label data/Phoenix-2014T/DGS_ex_labels.pkl --epochs 100 --batch_size 8

### for evaluation (if --asl_ex_label data/How2Sign/ASL_ex_labels.pkl for multilingual)
python main.py --function evaluate --asl_label data/How2Sign/labels.test --asl_dir How2Sign --asl_tok __ASL__token.pt --resume output/checkpoint.pt

## Model

The checkpoints are available for download.

Tokenization:
https://drive.google.com/file/d/1ZeHSGznXd_Xb285zZjNWuliMYae5EJbu
https://drive.google.com/file/d/1GYQNXTpok0bGikVGVv22z7EM-MksYpK-
https://drive.google.com/file/d/1h_2x4oNL8rAP-C1QXnOZjYx5_r0N7R2N

Fine-tuned Model:
https://drive.google.com/file/d/1ssoRsDVrYKd7qXjRd46u2X3sjYall00X
