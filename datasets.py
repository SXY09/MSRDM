import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
from collections import defaultdict
from transformers import PreTrainedTokenizer, AutoTokenizer
from utils import create_graph, Collator
from torch.utils.data import DataLoader
from utils import gen_dataset_coref
import spacy

cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}

class DocRED(Dataset):
    def __init__(self, data_module, dataset_dir: str, file_name: str, tokenizer: PreTrainedTokenizer,
                 force_regeneration: bool = False, use_coref: bool = True):
        # 构造加载元数据文件的函数
        super(DocRED, self).__init__()
        self.data_module = data_module
        self.name = "re-docred"
        dataset_dir = Path(dataset_dir)
        save_dir = dataset_dir / "bin"
        meta_dir = dataset_dir / "meta"
        # kg_dir = dataset_dir / "kg"
        with open(dataset_dir / "rel2id.json", "r", encoding="utf-8") as f:
            self.rel2id: Dict[str, int] = json.load(f)
        with open(dataset_dir / "ner2id.json", "r", encoding="utf-8") as f:
            self.ner2id: Dict[str, int] = json.load(f)
        self.id2rel = {value: key for key, value in self.rel2id.items()}
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_name_or_path = tokenizer.name_or_path
        model_name_or_path = model_name_or_path[model_name_or_path.rfind("/") + 1:]
        if use_coref:
            ori_path = gen_dataset_coref(self.data_module.coref_nlp, dataset_dir, file_name, force_regeneration)
        else:
            ori_path = str(dataset_dir / file_name)
        with open(ori_path, "r", encoding='utf-8') as fh:
            self.data: List[Dict] = json.load(fh)
        split = ori_path[ori_path.rfind("/") + 1:ori_path.rfind(".")]
        # kg_path = kg_dir / (file_name[:file_name.rfind('.')] + "_graph.json")
        # with open(kg_path, "r") as fh:
        #     self.kg: List[List[Dict]] = json.load(fh)
        save_path = save_dir / (split + f".{model_name_or_path}.pt")
        if os.path.exists(save_path) and not force_regeneration:
            print(f"Loading CDR {split} features ...")
            self.features = torch.load(save_path)
        else:
            self.features = self.read_docred(split, tokenizer)
            torch.save(self.features, save_path)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    # 从原始数据中读取文档、句子、实体提及和标签，进行实例化处理，并构建图结构
    def read_docred(self, split, tokenizer):
        i_line = 0
        pos_samples = 0
        neg_samples = 0
        features = []

        entity_1 = []
        entity_2 = []
        entity_3 = []
        entity_4 = []

        max_tokens_len = 0

        for docid, doc in tqdm(enumerate(self.data), desc=f"Reading CDR {split} data", total=len(self.data), ncols=100):
            delete_doc = [17222831, 11318962, 12073281, 17559688, 18312663, 20619739, 12398019, 18385788, 11911406]
            if doc['title'] in delete_doc:
                continue
            title: str = doc['title']
            entities: List[List[Dict]] = doc['vertexSet']
            sentences: List[List[str]] = doc['sents']
            ori_labels: List[Dict] = doc.get('labels', [])

            ENT_NUM = len(entities)
            SENT_NUM = len(sentences)
            MEN_NUM = len([m for e in entities for m in e if "coref" not in m])
            COREF_NUM = len([m for e in entities for m in e if "coref" in m])

            #处理全局pos
            # entity_start, entity_end = [], []
            # for entity in entities:
            #     for mention in entity:
            #         if "coref" not in mention:
            #             sent_id: int = mention["sent_id"]
            #             pos: List[int] = mention["pos"] #cdr是全局pos,左闭右闭
            #             entity_start.append((sent_id, pos[0]))
            #             entity_end.append((sent_id, pos[1] - 1))
            # assert len(entity_start) == len(entity_end) == MEN_NUM
            # entity_start = set(entity_star

            #对句子分词
            mention_index = []
            for entity in entities:
                for mention in entity:
                    if "coref" not in mention:
                       mention_index.append((mention["pos"][0], mention["pos"][1]))
            tokens: List[str] = [] #存放分词后的所有tokens
            # word2token: List[List[int]] = []
            sent_pos = {}
            sent_map = {}
            i_s = 0
            i_t = 0
            for sent in sentences:
                # idx_map = [0] * len(sent)  #长度为当前句子长度的列表，列表索引对应单词pos
                sent_pos[i_s] = len(tokens)
                for word in sent:
                    # idx_map[i_w] = len(tokens) #记录每个单词在分词后的索引，idx_map[word_pos] = new_word_pos
                    word_tokens = tokenizer.tokenize(word)
                    for start, end in mention_index:
                        if start == i_t:
                            word_tokens = ["*"] + word_tokens
                        if end == i_t + 1:
                            word_tokens = word_tokens + ["*"]
                    # if (i_s, i_w) in entity_start:
                    #     word_tokens = ["*"] + word_tokens
                    # if (i_s, i_w) in entity_end:
                    #     word_tokens = word_tokens + ["*"]
                    sent_map[i_t] = len(tokens)
                    tokens.extend(word_tokens)  #分词后的所有tokens内容
                    i_t += 1
                sent_map[i_t] = len(tokens) #存放原始token对应分词后的开始位置
                i_s += 1
                # idx_map.append(len(tokens)) #长度原始句子长度+1, 记录当前句子每个token对应分词后的开始位置，
                # word2token.append(idx_map)
            sent_pos[i_s] = len(tokens)

            #取处理后的句子开始结束位置
            final_sent_pos = []
            for i in range(len(sent_pos)-1):
                final_sent_pos.append((sent_pos[i], sent_pos[i+1]))

            # sent_pos = [(word2token[i][0], word2token[i][-1]) for i in range(SENT_NUM)]

            train_triple = {}
            for label in ori_labels:
                # h, t, r, evi = label['h'], label['t'], self.rel2id[label['r']], label['evidence']
                # train_triple[h, t].append({'relation': r, "evidence": evi})
                h, t, r = label['h'], label['t'], label['r']
                dist = label['dist']
                if (label['h'], label['t']) not in train_triple:
                    # train_triple[(label['h'], label['t'])] = [{'relation': r, 'evidence': evidence}]
                    train_triple[(label['h'], label['t'])] = [{'relation': r, 'dist': dist}]
                else:
                    # train_triple[(label['h'], label['t'])].append({'relation': r, 'evidence': evidence})
                    train_triple[(label['h'], label['t'])] = [{'relation': r, 'dist': dist}]

            coref_pos = [[] for _ in range(ENT_NUM)]
            entity_pos = [[] for _ in range(ENT_NUM)]
            mention_pos: Tuple[List[int], List[int]] = ([], [])

            entity_types: List[int] = []

            ent2mention: List[List[int]] = [[] for _ in range(ENT_NUM)]
            mention2ent: List[int] = []

            sent2mention: List[List[int]] = [[] for _ in range(SENT_NUM)]
            mention2sent: List[int] = []

            mention_id = 0
            entity_types = [0, 1]
            for entity_id, entity in enumerate(entities):
                # name_lens = np.array([len(m['name']) for m in entity if "coref" not in m])
                # long_idx = np.argmax(name_lens)
                # entity_types.append(self.ner2id[entity[long_idx]['type']])
                for mention in entity:
                    sent_id, pos = mention["sent_id"], mention["pos"]
                    # start = word2token[sent_id][pos[0]]
                    # end = word2token[sent_id][pos[1]]
                    start, end = pos[0], pos[1] #需要改为分词后的pos
                    new_start = sent_map[start]
                    new_end = sent_map[end]

                    if "coref" in mention:
                        coref_pos[entity_id].append((new_start, new_end))
                        continue

                    entity_pos[entity_id].append((new_start, new_end))
                    mention_pos[0].append(new_start)
                    mention_pos[1].append(new_end)

                    ent2mention[entity_id].append(mention_id)
                    mention2ent.append(entity_id)

                    sent2mention[sent_id].append(mention_id)
                    mention2sent.append(sent_id)

                    mention_id += 1
            assert sum(len(x) for x in coref_pos) == COREF_NUM

            hts: List[List[int]] = []
            relations: List[List[int]] = []
            dists, ent_dis = [], []
            for (h, t) in train_triple.keys():
                head_entity_pos, tail_entity_pos = entities[h][0]['pos'], entities[t][0]['pos']
                if head_entity_pos[1] < tail_entity_pos[0]:
                    abs_dis = tail_entity_pos[0] - head_entity_pos[1]
                    # abs_dis_id = distance_mapping[abs_dis]
                    # dis_id = 10 + abs_dis_id
                elif head_entity_pos[0] > tail_entity_pos[1]:
                    abs_dis = head_entity_pos[0] - tail_entity_pos[1]
                    # abs_dis_id = distance_mapping[abs_dis]
                    # dis_id = 10 - abs_dis_id
                else:
                    abs_dis = 0
                relation = [0] * len(gda_rel2id)
                for label in train_triple[h, t]:
                    r = label["relation"]
                    relation[r] = 1
                    if label["dist"] == "CROSS":
                        dist = 1
                    elif label["dist"] == "NON-CROSS":
                        dist = 0
                relations.append(relation)
                hts.append([h, t])
                dists.append(dist)
                ent_dis.append(abs_dis)
                pos_samples += 1

            max_tokens_len = max(max_tokens_len, len(tokens) + 2)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
            assert len(input_ids) == len(tokens) + 2

            entity_start, entity_end = {}, {}
            for start, end in mention_index:
                entity_start[(start)] = "*"
                entity_end[(end)] = "*"

            words = []
            lengthofPice = 0
            token_map = []
            for i_s, sent in enumerate(sentences):
                for i_t, token in enumerate(sent):
                    oneToken = []
                    words.append(token)
                    tokens_wordpiece = tokenizer.tokenize(token)
                    if any(i_t == item[0] for item in mention_index):
                        tokens_wordpiece = [entity_start[(i_t)]] + tokens_wordpiece
                        oneToken.append(lengthofPice + 1)
                        lengthofPice += len(tokens_wordpiece)
                        oneToken.append(lengthofPice)

                    elif any(i_t == item[1] for item in mention_index):
                        tokens_wordpiece = tokens_wordpiece + [entity_end[(i_t)]]
                        oneToken.append(lengthofPice)
                        lengthofPice += len(tokens_wordpiece)
                        oneToken.append(lengthofPice - 1)
                    else:
                        oneToken.append(lengthofPice)
                        lengthofPice += len(tokens_wordpiece)
                        oneToken.append(lengthofPice)
                    token_map.append(oneToken)

            # 图结构构建
            men_graph = create_graph(mention2ent, ent2mention, sent2mention, mention2sent,
                                                                self.rel2id, MEN_NUM, ENT_NUM, SENT_NUM,
                                                                1)

            i_line += 1
            feature = {
                'title': title,
                'input_ids': input_ids,
                'hts': hts,
                'sent_pos': final_sent_pos,
                'entity_pos': entity_pos,
                'coref_pos': coref_pos,
                'mention_pos': mention_pos[0],
                'entity_types': entity_types,
                'men_graph': men_graph,
                'label': relations,
                'dists': dists,
                'ent_dis': ent_dis,
            }
            features.append(feature)

            #if 1<=len(entity_pos)<5:
            #    entity_1.append(doc)
            #elif 5<=len(entity_pos)<10:
            #    entity_2.append(doc)
            #elif 10<= len(entity_pos)<15:
            #    entity_3.append(doc)
            #else:
            #    entity_4.append(doc)

        #print(len(entity_1),len(entity_2),len(entity_3),len(entity_4))
        #with open("F:\GDA_json格式/first.json", "w") as fh:
        #    json.dump(entity_1, fh)
        #with open("F:\GDA_json格式/second.json", "w") as fh:
        #    json.dump(entity_2, fh)
        #with open("F:\GDA_json格式/third.json", "w") as fh:
        #    json.dump(entity_3, fh)
        #with open("F:\GDA_json格式/fourth.json", "w") as fh:
        #    json.dump(entity_4, fh)

        print("# of documents {}.".format(i_line))
        print("maximum tokens length:", max_tokens_len)
        print("# of positive examples {}.".format(pos_samples))
        print("# of negative examples {}.".format(neg_samples))
        return features

# 封装数据集，tokenizer和dataloader，提供给模型用于 train、dev、test
class DocREDataModule:
    def __init__(
            self,
            dataset_dir: str,
            tokenizer: PreTrainedTokenizer,
            train_file: str,
            # train_distant_file: str,
            dev_file: str,
            test_file: str,
            force_regeneration: bool,
            use_coref: bool,
            train_batch_size: int,
            test_batch_size: int
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.train_file = train_file
        # self.train_distant_file = train_distant_file

        self.collate_fnt = Collator(tokenizer)  # collator对数据进行批量化打包传给模型进行训练等操作
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.coref_nlp = spacy.load("en_coreference_web_trf")

        self.data_train = DocRED(self, dataset_dir, train_file, tokenizer, force_regeneration, use_coref)

        self.data_dev = DocRED(self, dataset_dir, dev_file, tokenizer, force_regeneration, use_coref)

        self.data_test = DocRED(self, dataset_dir, test_file, tokenizer, force_regeneration, use_coref)

    @property
    def train_dataset(self):
        return self.data_train

    @property
    def dev_dataset(self):
        return self.data_dev

    @property
    def test_dataset(self):
        return self.data_test
    # dataloader对数据进行预处理
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fnt,
        )

    def dev_dataloader(self):
        return DataLoader(
            dataset=self.data_dev,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.collate_fnt,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.collate_fnt,
        )




if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('./PLM/deberta-v3-large')
    dm = DocREDataModule('./data/CDR', tokenizer, 'train.json', 'dev.json', 'test.json', force_regeneration= False, use_coref= False, test_batch_size=2, train_batch_size= False)
    #   dm = DocREDataModule('./data/DocRED', tokenizer, 'train_revised.json', 'dev_revised.json', 'test_revised.json', False, False, False, 2)
    # dm.gen_train_facts()
    # dm.data_train.official_evaluate_benchmark(torch.tensor([]))
