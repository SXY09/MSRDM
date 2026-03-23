import os
import shutil

import math
import json
import argparse
from collections import defaultdict
from copy import deepcopy
import logging

#import rich
import torch
import numpy as np
from tqdm import tqdm
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW

from utils import set_seed,assign_distance_bucket
import hydra
from utils import get_lr, print_config_tree

log = logging.getLogger(__name__)


def train(cfg, datamodule, model):
    args = cfg.train
    if args.seed:  #  
        set_seed(args.seed)
    model.to(args.device)

    # 加载训练、验证和测试数据集
    train_dataset, train_dataloader = datamodule.train_dataset, datamodule.train_dataloader()
    dev_dataset, dev_dataloader = datamodule.dev_dataset, datamodule.dev_dataloader()
    test_dataset, test_dataloader = datamodule.test_dataset, datamodule.test_dataloader()

    total_steps = args.epochs * (len(train_dataloader) // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"Total steps: {total_steps} = {args.epochs} epoch * ({len(train_dataloader)} batch // {args.gradient_accumulation_steps})")
    print(f"Warmup steps: {warmup_steps} = {total_steps} total steps * {args.warmup_ratio} warmup ratio")

    new_layer = ["extractor", "projection", "classifier", "conv"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)],
         "lr": args.classifier_lr},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # 梯度衰减策略 cos曲线或者线性衰减
    if args.lr_schedule == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = amp.GradScaler()

    num_steps = 0
    dev_best_score = -1
    test_best_score = -1
    model_name_or_path = cfg.model.model_name_or_path
    model_name_or_path = model_name_or_path[model_name_or_path.rfind("/") + 1:]
    for epoch in range(args.epochs):
        print("epoch: " + str(epoch))
        optimizer.zero_grad()
        for step, batch in enumerate(train_dataloader):
            model.train()
            inputs = {
                'input_ids': batch['input_ids'].to(args.device),
                'attention_mask': batch['attention_mask'].to(args.device),
                'hts': batch['hts'],
                'sent_pos': batch['sent_pos'],
                'entity_pos': batch['entity_pos'],
                'coref_pos': batch['coref_pos'],
                'mention_pos': batch['mention_pos'],
                'entity_types': batch['entity_types'],
                'men_graphs': batch['men_graphs'].to(args.device),
                'labels': batch['labels'],
            }
            # with torch.autograd.set_detect_anomaly(True):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = model(**inputs)
                # if epoch == 0 and step == 0 and loss > 5.5:
                #     print("initial loss: ", loss)
                #     print("Bad loss, Stop Training ...")  #
                #     return
                loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                num_steps += 1
            if (args.log_steps > 0 and step % args.log_steps == 0) or (step + 1 == len(train_dataloader)):
                print(f"{epoch}/{step}/{len(train_dataloader)}: current loss {round(loss.item(), 4)}")
            if (step + 1) == len(train_dataloader) \
                    or (args.evaluation_steps > 0
                        and num_steps > total_steps // 2
                        and num_steps % args.evaluation_steps == 0
                        and step % args.gradient_accumulation_steps == 0
                        and num_steps > args.start_steps):
                dev_score, dev_output = evaluate(cfg, model, dev_dataset, dev_dataloader, tag="dev")
                test_score, test_output = evaluate(cfg, model, test_dataset, test_dataloader,tag="test")
                print(dev_output)
                print(test_output)
                #if epoch == 0 and (step + 1) == len(train_dataloader) and dev_score < 35:  #
                #    print("Bad result, Stop Training ...")
                #    return
                lm_lr, classifier_lr = get_lr(optimizer)
                print(f'Current Step: {num_steps}, Current PLM lr: {lm_lr}, Current Classifier lr: {classifier_lr}')

                #if dev_score > dev_best_score or dev_score > 60:
                #    dev_best_score = dev_score
                #    test_score, test_output = evaluate(cfg, model, test_dataset, test_dataloader, tag="test")
                #    print(test_output)
                #    if test_score > test_best_score:
                #        test_best_score = test_score
                #        save_dir = args.save_best_path
                #        if not os.path.exists(save_dir):
                #            os.makedirs(save_dir)
                #        pre_max_model = [saved_model_name for saved_model_name in os.listdir(save_dir) if
                #                         saved_model_name[:saved_model_name.find('_')] == model_name_or_path]
                #        if len(pre_max_model) == 0:
                #            pre_max_score = -1
                #        else:
                #            pre_max_score = max(float(saved_model_name[saved_model_name.rfind('_') + 1:])
                #                                for saved_model_name in pre_max_model)
                #        if args.save_best_path and test_score > pre_max_score:
                #            sub_save_dir = f"{save_dir}/{model_name_or_path}_{round(test_score, 2)}"
                #            save_model_path = f"{sub_save_dir}/cdr_model.pth"
                #            save_config_path = f"{sub_save_dir}/config.txt"
                #            if not os.path.exists(sub_save_dir):
                #               os.makedirs(sub_save_dir)
                #            torch.save(model.state_dict(), save_model_path)
                #            print_config_tree(cfg, open(save_config_path, "w"))
                #            if pre_max_score != -1:
                #               shutil.rmtree(f"{save_dir}/{model_name_or_path}_{pre_max_score}")
                #if args.save_last_path:
                    #torch.save(model.state_dict(), args.save_last_path)


def evaluate(cfg, model, dataset, dataloader, tag="dev"):
    assert tag in {"dev", "test"}
    args = cfg.train

    if tag == "dev":
        print("Evaluating")
    else:
        print("Testing")
    preds, golds, dists, ent_dis = [], [], [], []
    id2rel = dataset.id2rel
    # rel_info = json.load(open("./data/CDR/meta/rel_info.json"))
    # dataset_kg_scores = []

    model.to(args.device)
    for batch in dataloader:
        model.eval()

        inputs = {
            'input_ids': batch['input_ids'].to(args.device),
            'attention_mask': batch['attention_mask'].to(args.device),
            'hts': batch['hts'],
            'sent_pos': batch['sent_pos'],
            'entity_pos': batch['entity_pos'],
            'coref_pos': batch['coref_pos'],
            'mention_pos': batch['mention_pos'],
            'entity_types': batch['entity_types'],
            'men_graphs': batch['men_graphs'].to(args.device),
            'labels': None,
        }

        with torch.no_grad():
            pred = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            golds.append(np.concatenate([np.array(label, np.float32) for label in batch['labels']], axis=0))
            dists.append(np.concatenate([np.array(dist, np.float32) for dist in batch['dists']], axis=0))
            ent_dis.append(np.concatenate([np.array(dist, np.float32) for dist in batch['ent_dis']], axis=0))

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    golds = np.concatenate(golds, axis=0).astype(np.float32)
    dists = np.concatenate(dists, axis=0).astype(np.float32)
    ent_dis = np.concatenate(ent_dis, axis=0).astype(np.float32)

    tp = ((preds[:, 1] == 1) & (golds[:, 1] == 1)).astype(np.float32).sum()  # 预测正确的
    fn = ((golds[:, 1] == 1) & (preds[:, 1] != 1)).astype(np.float32).sum()  # 正确但是预测错误的
    fp = ((preds[:, 1] == 1) & (golds[:, 1] != 1)).astype(np.float32).sum()  # 错误但是预测正确的
    tn = ((preds[:, 1] == 0) & (golds[:, 1] == 0)).astype(np.float32).sum()
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    fer = tn / (tn + fp + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)

    tp_intra = ((preds[:, 1] == 1) & (golds[:, 1] == 1) & (dists == 0)).astype(np.float32).sum()
    fn_intra = ((golds[:, 1] == 1) & (preds[:, 1] != 1) & (dists == 0)).astype(np.float32).sum()
    fp_intra = ((preds[:, 1] == 1) & (golds[:, 1] != 1) & (dists == 0)).astype(np.float32).sum()
    precision_intra = tp_intra / (tp_intra + fp_intra + 1e-5)
    recall_intra = tp_intra / (tp_intra + fn_intra + 1e-5)
    f1_intra = 2 * precision_intra * recall_intra / (precision_intra + recall_intra + 1e-5)

    tp_inter = ((preds[:, 1] == 1) & (golds[:, 1] == 1) & (dists == 1)).astype(np.float32).sum()
    fn_inter = ((golds[:, 1] == 1) & (preds[:, 1] != 1) & (dists == 1)).astype(np.float32).sum()
    fp_inter = ((preds[:, 1] == 1) & (golds[:, 1] != 1) & (dists == 1)).astype(np.float32).sum()
    precision_inter = tp_inter / (tp_inter + fp_inter + 1e-5)
    recall_inter = tp_inter / (tp_inter + fn_inter + 1e-5)
    f1_inter = 2 * precision_inter * recall_inter / (precision_inter + recall_inter + 1e-5)

    # 根据实体距离，检查模型性能，根据每个实体的第一个提及出现位置之间距离来将数据集分成不同的子集
    # 示例边界值，根据需求定义不同的距离区间
    distance_buckets = [8, 32, 64, 128]
    # 为每个样本的距离分配区间标签
    bucket_labels = np.array([assign_distance_bucket(d, distance_buckets) for d in ent_dis])

    # 然后基于这些标签来统计每个区间的TP, FP, FN
    buckets = len(distance_buckets) + 1  # 区间数+1，最后一个区间是>=最大边界值
    dis_tp = np.zeros(buckets, dtype=np.float32)
    dis_fp = np.zeros(buckets, dtype=np.float32)
    dis_fn = np.zeros(buckets, dtype=np.float32)

    for pred, gold, bucket in zip(preds[:, 1], golds[:, 1], bucket_labels):
        if pred == gold == 1:
            dis_tp[bucket] += 1
        elif pred == 1 and gold != 1:
            dis_fp[bucket] += 1
        elif pred != 1 and gold == 1:
            dis_fn[bucket] += 1
    dis_precision = dis_tp / (dis_tp + dis_fp + 1e-5)  # 防止分母为0
    dis_recall = dis_tp / (dis_tp + dis_fn + 1e-5)
    dis_f1_scores = 2 * dis_precision * dis_recall / (dis_precision + dis_recall + 1e-5)  # 防止分母为0

    output = {
        "{}_p".format(tag): precision * 100,
        "{}_r".format(tag): recall * 100,
        "{}_fer".format(tag): fer * 100,
        "{}_f1".format(tag): f1 * 100,
        "{}_f1_intra".format(tag): f1_intra * 100,
        "{}_f1_inter".format(tag): f1_inter * 100,
        "{}_f1_1".format(tag): dis_f1_scores[0] * 100,
        "{}_f1_2".format(tag): dis_f1_scores[1] * 100,
        "{}_f1_3".format(tag): dis_f1_scores[2] * 100,
        "{}_f1_4".format(tag): dis_f1_scores[3] * 100,
        "{}_f1_5".format(tag): dis_f1_scores[4] * 100,
    }
    return f1, output


@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.3")
def main(cfg):
    print_config_tree(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)

    log.info('Creating or Loading DataModule')
    datamodule = hydra.utils.instantiate(cfg.datamodule, tokenizer=tokenizer)()

    log.info("Creating DocRE Model")
    model = hydra.utils.instantiate(cfg.model, tokenizer=tokenizer)()

    if cfg.load_checkpoint:  # Training from checkpoint (for pre-training on distant dataset)
        log.info("Training from checkpoint")
        model.load_state_dict(torch.load(cfg.load_checkpoint))
        train(cfg, datamodule, model)
    elif cfg.load_path:  # Testing
        model.load_state_dict(torch.load(cfg.load_path))
        dev_score, dev_output = evaluate(cfg, model, datamodule.dev_dataset, datamodule.dev_dataloader(), tag="dev")
        print(dev_output)
        test_score, test_output = evaluate(cfg, model, datamodule.test_dataset, datamodule.test_dataloader(), tag="test")
        print(test_output)
    else:  # Training from scratch
        log.info("Training from scratch")
        train(cfg, datamodule, model)
    log.info("Finish Training or Testing")


if __name__ == "__main__":
    main()
