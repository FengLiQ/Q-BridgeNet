from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from config import get_args
from config import DEVICE, LOCAL_RANK, RANK, WORLD_SIZE
from config import FEATURE_POINT_SIZE
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch
import torch.nn as nn
from datasets import S2T_Dataset
from tqdm import tqdm
import sacrebleu
import os


class SignModel(nn.Module):
    def __init__(self, model_name, feat_dict):
        super(SignModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        self.feat_dict = feat_dict
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.pose_proj = nn.Sequential(
            nn.Linear(96, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
        )

    def save(self, path):
        torch.save(self.state_dict(), path)
        return

    def load(self, path):
        state_dict = torch.load(path)
        ret = self.load_state_dict(state_dict, strict=False)
        return ret

    def forward(self, x):
        embed_tokens = self.llm.get_input_embeddings()
        input_embeds_list = []
        label_list = []
        for key, prompt, ans in zip(x["key"], x["prompt"], x["ans"]):
            input_feat = self.feat_dict[key].unsqueeze(0)
            in_feat_embeds = self.pose_proj(input_feat)
            messages_sys = {"role": "system", "content": prompt}
            messages_user = {"role": "user", "content": ""}
            messages_assistant = {"role": "assistant", "content": ans}
            messages_sys_user_assistant_list = [messages_sys, messages_user, messages_assistant]
            messages_sys_user_list = [messages_sys, messages_user]

            sys_user_assistant = self.tokenizer.apply_chat_template(
                messages_sys_user_assistant_list,
                tokenize=True,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            ).to(DEVICE)
            sys_user = self.tokenizer.apply_chat_template(
                messages_sys_user_list,
                tokenize=True,
                return_tensors="pt",
            ).to(DEVICE)
            sys_user_prompt = self.tokenizer.apply_chat_template(
                messages_sys_user_list,
                padding="longest",
                enable_thinking=False,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            ).to(DEVICE)
            sys_user_assistant_embed = embed_tokens(sys_user_assistant)
            sys_user_assistant_embed = torch.cat([sys_user_assistant_embed[:, :sys_user.shape[1] - 2], in_feat_embeds,
                                                  sys_user_assistant_embed[:, sys_user.shape[1] - 2:]], dim=1)
            labels = sys_user_assistant.clone()
            labels[:, :sys_user_prompt.shape[1]] = -100
            labels = torch.cat([labels[:, :sys_user.shape[1] - 2], labels.new_full(in_feat_embeds.shape[:2], -100),
                                labels[:, sys_user.shape[1] - 2:]], dim=1)
            pad = self.tokenizer(self.tokenizer.pad_token, return_tensors="pt")["input_ids"].view(1, -1).to(DEVICE)
            pad_embed = embed_tokens(pad)
            if sys_user_assistant_embed.shape[1] < 160:
                l = 160 - sys_user_assistant_embed.shape[1]
                sys_user_assistant_embed = torch.cat((sys_user_assistant_embed, pad_embed.repeat(1, l, 1)), dim=1)
                labels = torch.cat((labels, pad.repeat(1, l)), dim=1)
            sys_user_assistant_embed = sys_user_assistant_embed[:, :160]
            labels = labels[:, :160]
            labels[labels == self.tokenizer.pad_token_id] = -100
            input_embeds_list.append(sys_user_assistant_embed)
            label_list.append(labels)
        input_embeds_list = torch.cat(input_embeds_list)
        label_list = torch.cat(label_list)
        out = self.llm(inputs_embeds=input_embeds_list, labels=label_list, return_dict=True)
        loss = out['loss']
        return loss

    def translate(self, prompt, feat):
        with torch.no_grad():
            embed_tokens = self.llm.get_input_embeddings()
            feat = feat.unsqueeze(0)
            in_feat_embeds = self.pose_proj(feat)
            messages_sys = {"role": "system", "content": prompt}
            messages_user = {"role": "user", "content": ""}
            messages_sys_user_list = [messages_sys, messages_user]

            sys_user = self.tokenizer.apply_chat_template(
                messages_sys_user_list,
                tokenize=True,
                return_tensors="pt",
            ).to(DEVICE)

            sys_user_prompt = self.tokenizer.apply_chat_template(
                messages_sys_user_list,
                padding="longest",
                enable_thinking=False,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            ).to(DEVICE)
            sys_user_prompt_embed = embed_tokens(sys_user_prompt)
            sys_user_prompt_embed = torch.cat([sys_user_prompt_embed[:, :sys_user.shape[1] - 2], in_feat_embeds,
                                               sys_user_prompt_embed[:, sys_user.shape[1] - 2:]], dim=1)

            outputs = self.llm.generate(
                inputs_embeds=sys_user_prompt_embed,
                max_new_tokens=256,
                do_sample=False,
                num_beams=4,
                eos_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def llm_data_compose(sign_data, sl_name, tok_model, output_feat_dict, output_data_list):
    def data_append(data_list, prompt, key, ans):
        data_list.append({"prompt": prompt, "key": key, "ans": ans})
        if len(data_list) % 5000 == 0:
            print(len(data_list))
            print(data_list[-1])
        return

    for item in sign_data:
        key = item[0]
        sign_item = item[1]
        ex_text = sign_data.get_ex_text(key)
        seg = sign_data.get_seg(key)
        token_input = []
        start = 0
        for clip_len in seg:
            clip = sign_item[start:start + clip_len].view(-1, FEATURE_POINT_SIZE * 2)
            ext_feat4 = clip[-1].unsqueeze(0).repeat(3 - (clip_len - 1) % 4, 1)
            clip = torch.cat((clip, ext_feat4), dim=0).view(1, -1, FEATURE_POINT_SIZE * 2).to(DEVICE)
            z_t, z_q, idx0, idx1 = tok_model.encode(clip)
            token_input.append(z_q[0])
            start += clip_len
        output_feat_dict[key] = torch.cat(token_input).detach()
        if ex_text is None:
            if sl_name == "ASL":
                tar_lang = "English"
            elif sl_name == "CSL":
                tar_lang = "Chinese"
            elif sl_name == "DGS":
                tar_lang = "German"
            else:
                tar_lang = "English"
            prompt = f"Translate {sl_name} Sign Language to {tar_lang}."
            ans = item[2]
            data_append(output_data_list, prompt, key, ans)
        else:
            for tar_lang in ex_text:
                prompt = f"Translate {sl_name} Sign Language to {tar_lang}."
                ans = ex_text[tar_lang]
                data_append(output_data_list, prompt, key, ans)


def finetuning(arg):
    output_dir = arg.output
    os.makedirs(output_dir, exist_ok=True)
    model_name = arg.model_name
    batch_size = arg.batch_size
    num_epoch = arg.epochs

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

    feat_dict = {}
    finetune_data = []
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
        if load_seg is None:
            print(sl_name, "missing seg")
            continue
        if load_tok is None:
            print(sl_name, "missing tok")
            continue
        if load_ex_label is None:
            print(sl_name, "missing optional ex_label")

        sign_data = S2T_Dataset(label_path=load_label, pose_dirs=load_dir, online_load=True)
        sign_data.load_seg(load_seg)
        tok_model = torch.load(load_tok).to(DEVICE)
        if load_ex_label is not None:
            sign_data.load_ex_text(load_ex_label)
        llm_data_compose(sign_data, sl_name, tok_model, feat_dict, finetune_data)

    model = SignModel(model_name, feat_dict)
    model = model.to(DEVICE)
    if arg.resume is not None:
        print('***********************************')
        print('Load Checkpoint...')
        print('***********************************')
        ret = model.load(arg.resume)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))

    dist.barrier()
    model = DDP(model, device_ids=[LOCAL_RANK])
    print(model)

    dist.barrier()

    train_sampler = DistributedSampler(
        finetune_data,
        num_replicas=WORLD_SIZE,
        rank=RANK,
        shuffle=True
    )
    dataloader = DataLoader(
        finetune_data,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epoch):
        train_sampler.set_epoch(epoch)
        model.train()
        print(epoch)
        pbar = tqdm(dataloader, disable=(RANK != 0))
        for batch in pbar:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            if RANK == 0:
                pbar.set_description(f"loss {loss.item():.4f}")
            last_key = batch["key"][0]
            last_prompt = batch["prompt"][0]
            last_gt = batch["ans"][0]

        dist.barrier()
        if RANK == 0:
            print(loss)
            print(last_key)
            print(last_prompt)
            print(last_gt)
        if RANK == 0:
            model.module.save(os.path.join(output_dir, "checkpoint" + str(epoch) + ".pt"))
        dist.barrier()

    if RANK == 0:
        model.module.save(os.path.join(output_dir, "checkpoint.pt"))


if __name__ == '__main__':
    args = get_args()
    finetuning(args)
