from config import get_args
from datasets import S2T_Dataset
import torch
from config import DEVICE, TEMP_DIR
from segtok import segment
import os
from finetuning import SignModel, llm_data_compose
from torch.utils.data import DataLoader
from tqdm import tqdm
from rouge import Rouge
import sacrebleu
import pickle
import numpy as np


def evaluation(arg):
    output_dir = arg.output
    os.makedirs(output_dir, exist_ok=True)
    model_name = arg.model_name

    asl_label = arg.asl_label
    asl_dir = arg.asl_dir
    asl_seg = arg.asl_seg
    asl_tok = arg.asl_tok
    asl_ex_label = arg.asl_ex_label
    csl_label = arg.csl_label
    csl_dir = arg.csl_dir
    csl_seg = arg.csl_seg
    csl_tok = arg.csl_tok
    csl_ex_label = arg.csl_ex_label
    dgs_label = arg.dgs_label
    dgs_dir = arg.dgs_dir
    dgs_seg = arg.dgs_seg
    dgs_tok = arg.dgs_tok
    dgs_ex_label = arg.dgs_ex_label

    sign_label = ["ASL", "CSL", "DGS"]
    for i, sl_name in enumerate(sign_label):
        if sl_name == "ASL":
            load_label = asl_label
            load_dir = asl_dir
            load_seg = asl_seg
            load_tok = asl_tok
            load_ex_label = asl_ex_label
        elif sl_name == "CSL":
            load_label = csl_label
            load_dir = csl_dir
            load_seg = csl_seg
            load_tok = csl_tok
            load_ex_label = csl_ex_label
        elif sl_name == "DGS":
            load_label = dgs_label
            load_dir = dgs_dir
            load_seg = dgs_seg
            load_tok = dgs_tok
            load_ex_label = dgs_ex_label
        else:
            continue
        if load_label is None:
            print(sl_name, "missing label")
            continue
        if load_dir is None:
            print(sl_name, "missing dir")
            continue
        if load_tok is None:
            print(sl_name, "missing tok")
            continue
        if load_ex_label is None:
            print(sl_name, "missing optional ex_label")

        sign_data = S2T_Dataset(label_path=load_label, pose_dirs=load_dir, online_load=True)
        tok_model = torch.load(load_tok).to(DEVICE)
        if load_ex_label is not None:
            sign_data.load_ex_text(load_ex_label)
        if load_seg is None:
            print("try seg")
            segment(sign_data, tok_model, os.path.join(TEMP_DIR, sl_name + "_eval.pkl"))
        else:
            print("load seg")
            sign_data.load_seg(load_seg)
        feat_dict = {}
        test_data = []
        llm_data_compose(sign_data, sl_name, tok_model, feat_dict, test_data)

        model = SignModel(model_name, feat_dict)
        model = model.to(DEVICE)
        print('***********************************')
        print('Load Checkpoint...')
        print('***********************************')
        ret = model.load(arg.resume)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))

        rouge = Rouge()
        translation_dict = {}
        ans_dict = {}
        tar_lang_list = ["English", "Chinese", "German"]
        for tar_lang in tar_lang_list:
            translation_dict[tar_lang] = []
            ans_dict[tar_lang] = []

        dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
        pbar = tqdm(dataloader)
        for batch in pbar:
            key = batch["key"][0]
            prompt = batch["prompt"][0]
            ans = batch["ans"][0]
            feat = feat_dict[key]
            tar_lang = "English"
            for lang in tar_lang_list:
                if lang in prompt:
                    tar_lang = lang
            translation = model.translate(prompt, feat)
            translation_dict[tar_lang].append(translation)
            ans_dict[tar_lang].append(ans)

        b_4 = {}
        b_1 = {}
        rg = {}
        for tar_lang in tar_lang_list:
            if len(translation_dict[tar_lang]) == 0:
                continue
            translation = translation_dict[tar_lang]
            ans = ans_dict[tar_lang]
            if tar_lang == "Chinese":
                bl = sacrebleu.corpus_bleu(translation, [ans], tokenize="zh")
                trans_result = [" ".join(x) for x in translation]
                ans_result = [" ".join(x) for x in ans]
                r = rouge.get_scores(trans_result, ans_result, avg=True)
            else:
                bl = sacrebleu.corpus_bleu(translation, [ans])
                r = rouge.get_scores(translation, ans, avg=True)
            b_4[tar_lang] = bl.scores[3]
            b_1[tar_lang] = bl.scores[0]
            rg[tar_lang] = r["rouge-1"]["f"]
        print(sl_name)
        print("BLEU-4", np.mean(list(b_4.values())))
        print("BLEU-1", np.mean(list(b_1.values())))
        print("ROUGE", np.mean(list(rg.values())))    
        

if __name__ == '__main__':
    args = get_args()
    evaluation(args)
