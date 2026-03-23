import json
import os

import torch
import random
import numpy as np
import dgl
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import rich
import rich.syntax
import rich.tree
from collections import defaultdict, Counter


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(optimizer):
    lm_lr = optimizer.param_groups[0]['lr']
    classifier_lr = optimizer.param_groups[1]['lr']
    return lm_lr, classifier_lr


def print_config_tree(cfg: DictConfig, file=None):
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    for filed in cfg:
        branch = tree.add(filed, style=style, guide_style=style)
        config_group = cfg[filed]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=False)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree, file=file)

# 处理批处理，将不同字段合并为模型所需的输入格式
class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        titles = [f['title'] for f in batch]
        input_ids = {'input_ids': [f['input_ids'] for f in batch]}
        hts = [f["hts"] for f in batch]
        sent_pos = [f['sent_pos'] for f in batch]
        entity_pos = [f["entity_pos"] for f in batch]
        coref_pos = [f['coref_pos'] for f in batch]
        mention_pos = [torch.tensor(f["mention_pos"]) for f in batch]
        entity_types = [f['entity_types'] for f in batch]
        men_graphs = dgl.batch([f['men_graph'] for f in batch])
        # ent_graphs = dgl.batch([f['ent_graph'] for f in batch])
        # etypes = torch.cat([f['etype'] for f in batch])
        # e_labels = torch.cat([f['e_label'] for f in batch])
        edge_hts = []
        # for f in batch:
        #     z = f['ent_graph'].all_edges()
        #     edge_hts.append(list(zip(z[0].tolist(), z[1].tolist())))
        labels = [f["label"] for f in batch]
        dists = [f["dists"] for f in batch]
        ent_dis = [f["ent_dis"] for f in batch]

        inputs = self.tokenizer.pad(input_ids, return_tensors='pt')
        output = {
            "titles": titles,
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "hts": hts,
            "sent_pos": sent_pos,
            "entity_pos": entity_pos,
            "coref_pos": coref_pos,
            "mention_pos": mention_pos,
            "entity_types": entity_types,
            "men_graphs": men_graphs,
            # "ent_graphs": ent_graphs,
            # "etypes": etypes,
            # "e_labels": e_labels,
            # "edge_hts": edge_hts,
            "labels": labels,
            "dists": dists,
            "ent_dis": ent_dis,
        }
        return output

# 利用DGL进行异构图构建，实体、句子、文档三种异构节点，从而增强实体表示
def create_graph(men2ent, ent2men, sent2men, men2sent, rel2id, MEN_NUM, ENT_NUM, SENT_NUM, DOC_NUM):
    men_graph_dict = {}

    doc_id = list(range(DOC_NUM))  # doc node
    men_ids = list(range(DOC_NUM, DOC_NUM + MEN_NUM))  # mention nodes
    sent_ids = list(range(DOC_NUM + MEN_NUM, DOC_NUM + MEN_NUM + SENT_NUM))  # sent nodes

    # ================================================doc--sent=========================================================
    men_graph_dict["node", "d-s", "node"] = (doc_id * SENT_NUM + sent_ids, sent_ids + doc_id * SENT_NUM)

    # ================================================sent--sent========================================================
    sss = []
    for i in range(SENT_NUM):
        for j in range(i + 1, SENT_NUM):
            sss.append((sent_ids[i], sent_ids[j]))
            sss.append((sent_ids[j], sent_ids[i]))
    men_graph_dict["node", "s-s", "node"] = sss

    # ===============================================mention--sent======================================================
    men2sent = np.array(sent_ids)[men2sent].tolist()
    men_graph_dict["node", "s-m", "node"] = (men_ids + men2sent, men2sent + men_ids)

    # ==========================================intra-entity-mention--mention===========================================
    ie_mms = []
    for ems in ent2men:
        n = len(ems)
        for i in range(n):
            for j in range(i + 1, n):
                x, y = ems[i], ems[j]
                ie_mms.append((men_ids[x], men_ids[y]))
                ie_mms.append((men_ids[y], men_ids[x]))
    men_graph_dict["node", "ie/m-m", "node"] = ie_mms

    # ============================================intra-sent-mention--mention===========================================
    is_mms = []
    for sms in sent2men:
        n = len(sms)
        for i in range(n):
            for j in range(i + 1, n):
                x, y = sms[i], sms[j]
                is_mms.append((men_ids[x], men_ids[y]))
                is_mms.append((men_ids[y], men_ids[x]))

    men_graph_dict["node", "is/m-m", "node"] = is_mms

    # =================================================knowledge-graph==================================================
    # u, v, etype = [], [], []
    # for ent_id, edges in enumerate(kg):
    #     for rel, objs in edges.items():
    #         for obj in objs:
    #             u.append(ent_id)
    #             v.append(obj)
    #             etype.append(rel2id[rel] - 1)
    #             # ent_graph_dict["node", rel2id[rel] - 1, "node"].append((ent_id, obj))
    # e_label = torch.zeros(len(etype))
    # for i, (s, t, r) in enumerate(zip(u, v, etype)):
    #     e_label[i] = true_label[s, t, r + 1]
    # # ent_graph = dgl.heterograph(ent_graph_dict, num_nodes_dict={"node": ENT_NUM + 1})
    # ent_graph = dgl.graph((u, v), num_nodes=ENT_NUM)
    # assert ent_graph.num_edges() == sum(len(edges) for node_dict in kg for edges in node_dict.values())
    # =================================================knowledge-graph==================================================

    men_graph = dgl.heterograph(men_graph_dict)
    assert men_graph.num_nodes() == DOC_NUM + MEN_NUM + SENT_NUM
    # assert ent_graph.num_nodes("men") == MEN_NUM
    # assert ent_graph.num_nodes("ent") == ENT_NUM

    assert men_graph.num_edges("d-s") == SENT_NUM * 2
    assert men_graph.num_edges("s-s") == SENT_NUM * (SENT_NUM - 1)
    assert men_graph.num_edges("s-m") == MEN_NUM * 2
    # assert ent_graph.num_edges('m->e') == MEN_NUM
    # assert ent_graph.num_edges() == sum(len(edges) for node_dict in kg for edges in node_dict.values()) + \
    #        len(rel2id) - 1 - len(set(k for d in kg for k in d.keys()))


    def fc_edge_nums(gms):
        edge_nums = 0
        for ms in gms:
            gn = len(ms)
            edge_nums += gn * (gn - 1)
        return edge_nums

    assert men_graph.num_edges("is/m-m") == fc_edge_nums(sent2men)
    assert men_graph.num_edges("ie/m-m") == fc_edge_nums(ent2men)
    # assert len(ent_graph.etypes) == len(rel2id) - 1

    # return men_graph, ent_graph, torch.tensor(etype), e_label
    return men_graph

def gen_coref(coref_nlp, doc_id, sample):
    sents = sample['sents']
    entities = sample['vertexSet']
    document = ''
    word2char = []
    word2sent = []
    sent2word = []
    word_cnt = 0
    for sent_id, sent in enumerate(sents):
        sent2word.append([])
        for word_id, word in enumerate(sent):
            word2char.append([])
            word2sent.append([sent_id, word_id])
            word2char[-1].append(len(document))
            document += word
            word2char[-1].append(len(document))
            document += ' '
            sent2word[-1].append(word_cnt)
            word_cnt += 1
    assert len(word2char) == len(word2sent) == sum(len(sent) for sent in sents) == word_cnt
    document = document[:-1]
    WORD_NUM, CHAR_NUM = len(word2char), len(document)
    char2word = np.array([-1] * CHAR_NUM)
    for word_id, (start_idx, end_idx) in enumerate(word2char):
        char2word[start_idx:end_idx] = word_id
    doc = coref_nlp(document)
    clusters = [val for key, val in doc.spans.items() if key.startswith("coref_cluster")]
    char2cluster = np.array([-1] * CHAR_NUM)
    for cluster_id, cluster in enumerate(clusters):
        for mention_span in cluster:
            span = char2cluster[mention_span.start_char:mention_span.end_char]
            span[span == -1] = cluster_id
    char2entity = np.array([-1] * CHAR_NUM)
    entity_clusters = defaultdict(Counter)
    for entity_id, entity in enumerate(entities):
        for mention_id, mention in enumerate(entity):
            sent_id, start_word, end_word = mention['sent_id'], mention['pos'][0], mention['pos'][1] - 1
            # start_word_idx, end_word_idx = sent2word[sent_id][start_word], sent2word[sent_id][end_word]
            # start_idx, end_idx = word2char[start_word_idx][0], word2char[end_word_idx][1]
            start_idx, end_idx = word2char[start_word][0], word2char[end_word][1]
            char2entity[start_idx:end_idx] = entity_id
            cluster_id = set(np.unique(char2cluster[start_idx:end_idx]))
            cluster_id.discard(-1)
            # assert len(cluster_id) <= 1, f"{doc_id}/{entity_id}/{mention_id}"
            if len(cluster_id) > 1 and entity_id in entity_clusters:
                del entity_clusters[entity_id]
                break
            if cluster_id:
                entity_clusters[entity_id][cluster_id.pop()] += 1
    entities = sample['vertexSet']
    for entity_id, entity_cluster in entity_clusters.items():
        # assert len(entity_cluster) == 1, f"{doc_id}/{entity_id}/{entity_cluster}"
        max_time = -1
        cluster_id = -1
        for k, v in entity_cluster.items():
            if v > max_time:
                cluster_id, max_time = k, v
        # cluster_id = entity_cluster.pop()
        cluster = clusters[cluster_id]
        for mention_span in cluster:
            if all(np.unique(char2entity[mention_span.start_char:mention_span.end_char]) == -1):
                word_ids = np.unique(char2word[mention_span.start_char:mention_span.end_char])
                word_ids = sorted(list(word_ids))
                if -1 in word_ids:
                    word_ids.pop(0)
                start_sent_id, start_word_id = word2sent[word_ids[0]]
                end_sent_id, end_word_id = word2sent[word_ids[-1]]
                # assert start_sent_id == end_sent_id, f"{doc_id}/{entity_id}/{mention_span.text}"
                if start_sent_id != end_sent_id:
                    continue
                entities[entity_id].append({
                    "sent_id": start_sent_id,
                    # "pos": [start_word_id, end_word_id + 1],
                    "pos": [start_word_id, end_word_id],
                    "name": document[mention_span.start_char:mention_span.end_char],
                    "type": entities[entity_id][0]['type'],
                    "coref": True
                })
    return sample


def gen_dataset_coref(coref_nlp, dataset_dir, filename, force_regeneration):
    split = filename[:filename.rfind(".")]
    save_path = os.path.join(dataset_dir, f"{split}_coref.json")
    if os.path.exists(save_path) and not force_regeneration:
        return save_path
    dataset = json.load(open(os.path.join(dataset_dir, filename)))
    for doc_id, sample in tqdm(enumerate(dataset), desc=f"gen {split} data coref", ncols=100, total=len(dataset)):
        gen_coref(coref_nlp, doc_id, sample)
    json.dump(dataset, open(save_path, "w"))
    return save_path

def assign_distance_bucket(distance, buckets):
    """根据距离分配区间标签"""
    for i, bound in enumerate(buckets):
        if distance < bound:
            return i
    return len(buckets)