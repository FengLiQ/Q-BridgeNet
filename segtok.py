from config import get_args
from datasets import S2T_Dataset, CLIP_Dataset4
from config import MAX_CLIP_LEN, LANG_LABEL, FEATURE_POINT_SIZE, DEVICE, TEMP_DIR
import torch
from showing import showimg
import time
import torch.nn as nn
from rvq import Encoder, Decoder, RVQ
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import gzip
import pickle


def segment(sign_data, token_model, output_path):
    seg_data = {}
    for idx, item in enumerate(sign_data):  # name_sample, pose_sample, pose_ori, text, gloss, seg
        name = item[0]
        sign_item = item[1]
        sign_item = sign_item.view(-1, FEATURE_POINT_SIZE * 2).to(DEVICE)
        f = torch.ones(len(sign_item) + 1) * 100000
        f_path = torch.zeros(len(sign_item) + 1).int()
        f[0] = 0
        for i in range(len(sign_item)):
            for length_idx in range(MAX_CLIP_LEN // 4):
                feat = []
                for j in range(1, 5):
                    length = length_idx * 4 + j
                    if i + 1 - length < 0:
                        break
                    clip = sign_item[i + 1 - length:i + 1]
                    ext_feat = sign_item[i].unsqueeze(0).repeat(4 - j, 1)
                    feat.append(torch.cat((clip, ext_feat), dim=0))
                if len(feat) == 0:
                    break
                feat = torch.stack(feat)
                rec_x, _, __ = token_model(feat)
                rec_loss = torch.nn.functional.mse_loss(feat.flatten(1), rec_x.flatten(1), reduction="none").sum(dim=-1)
                for j in range(1, 5):
                    length = length_idx * 4 + j
                    if i + 1 - length < 0:
                        break
                    z = f[i + 1 - length] + rec_loss[j - 1]
                    if z < f[i + 1]:
                        f[i + 1] = z
                        f_path[i + 1] = length
        i = len(sign_item)
        seg_length = []
        while i > 0:
            length = int(f_path[i])
            seg_length.append(length)
            i = i - length
        seg_length.reverse()
        seg_data[name] = seg_length
        if idx % 100 == 0:
            print(idx)
            print(item[2])
            print(seg_length)
            start = 0
            for clip_len in seg_length:
                showimg(item[1][start:start + clip_len].cpu(),
                        os.path.join(TEMP_DIR, str(time.time()) + item[0] + "_seg_"))
                start += clip_len

        if idx % 100 == 0 and idx != 0:
            with gzip.open(output_path, "wb") as f:
                pickle.dump(seg_data, f)

    with gzip.open(output_path, "wb") as f:
        pickle.dump(seg_data, f)
    sign_data.seg_data = seg_data


def tokenize(data_list, lang_list, output_dir="", batch_size=1024):
    os.makedirs(output_dir, exist_ok=True)
    var_len_clip = {}
    for i, sign_label in enumerate(lang_list):
        var_len_clip[sign_label] = {}
        for l in range(1, MAX_CLIP_LEN + 1):
            var_len_clip[sign_label][l] = []
        for item in data_list[i]:
            key = item[0]
            pose = item[1]
            seg = data_list[i].get_seg(key)
            start = 0
            for clip_len in seg:
                var_len_clip[sign_label][clip_len].append(pose[start:start + clip_len].view(-1, FEATURE_POINT_SIZE * 2))
                start += clip_len

    num_epoch = 256
    clip_dataset = {}
    for sign_label in lang_list:
        clip_dataset[sign_label] = {}
        for j in range(MAX_CLIP_LEN // 4):
            clip_dataset[sign_label][j] = CLIP_Dataset4(var_len_clip[sign_label], j, 0)

    num_embeddings_shared = 1024
    num_embeddings_private = 512
    embedding_dim = 96
    hidden_dim = 1024

    vq_embedding_shared = nn.Embedding(num_embeddings_shared, embedding_dim)
    vq_embedding_shared.weight.data.uniform_(-1 / num_embeddings_shared, 1 / num_embeddings_shared)
    encoder = Encoder(in_channels=FEATURE_POINT_SIZE * 2, hidden_dim=hidden_dim, out_channels=embedding_dim)
    decoder = Decoder(in_channels=embedding_dim, hidden_dim=hidden_dim, out_channels=FEATURE_POINT_SIZE * 2)
    vq_embedding_private = {}
    tok_model = {}
    for sign_label in lang_list:
        vq_embedding_private[sign_label] = nn.Embedding(num_embeddings_private, embedding_dim)
        vq_embedding_private[sign_label].weight.data.uniform_(-1 / num_embeddings_private, 1 / num_embeddings_private)
        tok_model[sign_label] = RVQ(encoder, vq_embedding_shared, vq_embedding_private[sign_label], decoder).to(DEVICE)

    param_list = list(encoder.parameters()) + list(decoder.parameters()) + list(vq_embedding_shared.parameters())
    for sign_label in lang_list:
        param_list = param_list + list(vq_embedding_private[sign_label].parameters())
    optimizer = torch.optim.AdamW(param_list, lr=1e-4)

    for epoch in range(num_epoch):
        print(epoch)
        for i in range(MAX_CLIP_LEN // 4):
            max_size = -1
            max_label = None
            for sign_label in lang_list:
                if len(clip_dataset[sign_label][i]) > max_size:
                    max_size = len(clip_dataset[sign_label][i])
                    max_label = sign_label

            dataloader = DataLoader(clip_dataset[max_label][i], batch_size=batch_size, shuffle=True, drop_last=True)
            pbar = tqdm(dataloader)
            for batch, idx in pbar:
                loss = 0
                train_info = {}
                last_batch = {}
                last_decoded = {}
                for sign_label in lang_list:
                    sign_batch = clip_dataset[sign_label][i].feat[idx % len(clip_dataset[sign_label][i])]
                    x_recon, recon_loss, vq_loss = tok_model[sign_label](sign_batch)
                    last_batch[sign_label] = sign_batch[0].view(-1, FEATURE_POINT_SIZE, 2)
                    last_decoded[sign_label] = x_recon[0].view(-1, FEATURE_POINT_SIZE, 2)
                    loss = loss + recon_loss + vq_loss
                    train_info["rec_loss" + sign_label] = recon_loss
                    train_info["vq_loss" + sign_label] = vq_loss
                train_info["total_loss"] = loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0 or epoch == num_epoch - 1:
                print(train_info)
            if epoch % 100 == 0 or epoch == num_epoch - 1:
                for sign_label in lang_list:
                    showimg(last_batch[sign_label].cpu(), os.path.join(TEMP_DIR, str(time.time()) + sign_label + "_real_"))
                    showimg(last_decoded[sign_label].detach().cpu(), os.path.join(TEMP_DIR, str(time.time()) + sign_label + "_REC_"))

        if epoch % 10 == 0 or epoch == num_epoch - 1:
            for sign_label in lang_list:
                torch.save(tok_model[sign_label], os.path.join(output_dir, sign_label + "token.pt"))
    return tok_model


def segtok(arg):
    asl_label = arg.asl_label
    asl_dir = arg.asl_dir
    asl_seg = arg.asl_seg
    csl_label = arg.csl_label
    csl_dir = arg.csl_dir
    csl_seg = arg.csl_seg
    dgs_label = arg.dgs_label
    dgs_dir = arg.dgs_dir
    dgs_seg = arg.dgs_seg
    epochs = arg.epochs
    output_dir = arg.output
    asl_data = S2T_Dataset(label_path=asl_label, pose_dirs=asl_dir)
    csl_data = S2T_Dataset(label_path=csl_label, pose_dirs=csl_dir)
    dgs_data = S2T_Dataset(label_path=dgs_label, pose_dirs=dgs_dir)

    if asl_seg is None:
        print("asl_seg uses ini_rand_seg")
        asl_data.ini_seg()
    else:
        asl_data.load_seg(asl_seg)

    if csl_seg is None:
        print("csl_seg uses ini_rand_seg")
        csl_data.ini_seg()
    else:
        csl_data.load_seg(csl_seg)

    if dgs_seg is None:
        print("dgs_seg uses ini_rand_seg")
        dgs_data.ini_seg()
    else:
        dgs_data.load_seg(dgs_seg)

    for epoch in range(epochs):
        tok_model = tokenize([asl_data, csl_data, dgs_data], LANG_LABEL, output_dir + str(epoch))
        segment(asl_data, tok_model["__ASL__"], os.path.join(output_dir + str(epoch), "__ASL__seg.pkl"))
        segment(csl_data, tok_model["__CSL__"], os.path.join(output_dir + str(epoch), "__CSL__seg.pkl"))
        segment(dgs_data, tok_model["__DGS__"], os.path.join(output_dir + str(epoch), "__DGS__seg.pkl"))
    tokenize([asl_data, csl_data, dgs_data], LANG_LABEL, output_dir + str(epochs))


if __name__ == '__main__':
    args = get_args()
    segtok(args)
