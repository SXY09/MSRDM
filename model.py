from typing import List, Tuple

#import hydra.utils
import math
import torch
import torch.nn as nn
from opt_einsum import contract
from torch import Tensor

from losses import AFLoss
from long_seq import process_long_input, process_long_input_longformer
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import RelGraphConv
from transformers import AutoModel, AutoConfig
import numpy as np
from dma import TokenTuringMachineEncoder
from torch.nn import Softmax

class GCNGraphConvLayer(nn.Module):

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 *,
                 weight=True,
                 bias=True,
                 activation=nn.Tanh(),
                 self_loop=True,
                 dropout=0.):
        super(GCNGraphConvLayer, self).__init__()
        num_bases = len(rel_names)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}
        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class GATGraphConvLayer(nn.Module):
    def __init__(self, in_feat, out_feat, rel_names, fp, ap, residual, activation):
        super(GATGraphConvLayer, self).__init__()
        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(in_feat, out_feat, num_heads=1, feat_drop=fp, attn_drop=ap, residual=residual, activation=activation)
            for rel in rel_names
        })

    def forward(self, g, inputs):
        hs = self.conv(g, inputs)
        return {ntype: h.squeeze(1) for ntype, h in hs.items()}


class GATGraphConv(nn.Module):
    def __init__(self, hidden_dim, edge_types, feat_drop, attn_drop, residual, activation, num_layers):
        super().__init__()
        self.graph_conv = nn.ModuleList([
            GATGraphConvLayer(hidden_dim, hidden_dim, edge_types, feat_drop, attn_drop, residual, activation)
            for _ in range(num_layers)
        ])

    def forward(self, graph, feat):
        for graph_layer in self.graph_conv:
            feat = graph_layer(graph, {'node': feat})['node']
        return feat


class NoGraphConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, feat):
        return torch.zeros_like(feat)

class CC_module(nn.Module):

    def __init__(self, in_dim):
        super(CC_module, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        # self.INF = self.INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def INF(self, B, H, W):
        return -torch.diag(torch.tensor(float("inf")).cuda(0).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,                                                                                                      1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,                                                                                                     1)

        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)

        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        # concate = concate * (concate>torch.mean(concate,dim=3,keepdim=True)).float()

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)

        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x


class DocREModel(nn.Module):
    def __init__(self,
                 model_name_or_path,
                 max_seq_length,
                 transformer_type,
                 tokenizer,
                 graph_conv,
                 residual,
                 coref,
                 num_class,
                 block_size,
                 reason_type,
                 ttm_re,
                 loss_fnt,
                 num_reasoning_layers):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.max_seq_length = max_seq_length
        self.config.cls_token_id = tokenizer.cls_token_id
        self.config.sep_token_id = tokenizer.sep_token_id
        self.config.transformer_type = transformer_type
        self.config.model_max_len = self.config.max_position_embeddings
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.hidden_size = self.config.hidden_size
        self.emb_size = self.hidden_size
        self.block_size = block_size
        self.rel_loss_fnt = loss_fnt
        self.head_extractor = nn.Linear(2 * self.hidden_size, self.emb_size)
        self.tail_extractor = nn.Linear(2 * self.hidden_size, self.emb_size)
        self.projection = nn.Linear(self.emb_size, self.hidden_size, bias=False)
        self.classifier = nn.Linear(self.hidden_size, num_class)
        if isinstance(graph_conv, NoGraphConv):   #  -w/o graph neural network
            self.graph_conv = graph_conv
        else:
            self.graph_conv = graph_conv(hidden_dim=self.hidden_size)
        self.residual = residual
        assert coref in {'gated', 'e_context', 'none'}
        self.coref = coref
        # self.emb_size = emb_size
        self.block_size = block_size
        self.num_class = num_class
        # self.num_labels = num_labels
        self.mu_encoder = TokenTuringMachineEncoder(process_size=4, memory_size=200, input_dim=self.emb_size,
                                                    mlp_dim=self.emb_size, num_layers=4)
        self.dropout = nn.Dropout(p=0.1)
        self.ttm_re = ttm_re
        #添加逻辑推理模块
        self.reason_type = reason_type
        self.inter_channel = int(self.emb_size // 2)
        self.conv_reason_e_l1 = nn.Sequential(
            nn.Conv2d(self.emb_size, self.inter_channel, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(inplace=True),
        )
        self.conv_reason_e_l2 = nn.Sequential(
            nn.Conv2d(self.inter_channel, self.inter_channel, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(inplace=True), )
        self.conv_reason_e_l3 = nn.Sequential(
            nn.Conv2d(self.inter_channel, self.emb_size, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(self.emb_size),
            nn.ReLU(inplace=True), )
        self.cc_module = CC_module(self.inter_channel)
        self.num_reasoning_layers = num_reasoning_layers
        if self.reason_type == 'criss-cross':
            self.reasoning_convs = nn.ModuleList()  # 存储卷积层
            self.reasoning_ccs = nn.ModuleList()  # 存储CC模块

            for i in range(self.num_reasoning_layers):
                # 逻辑：
                # 1. 输入维度：如果是第一层(i=0)，输入是 emb_size (768)；否则是 inter_channel (384)
                # 2. 输出维度：如果是最后一层(i=L-1)，输出回 emb_size (768)；否则保持 inter_channel (384)

                in_channels = self.emb_size if i == 0 else self.inter_channel
                out_channels = self.emb_size if i == self.num_reasoning_layers - 1 else self.inter_channel

                # 定义卷积块: Conv -> BN -> ReLU
                conv_layer = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )

                # 定义该层的 CC 模块 (需要匹配该层的输出维度)
                cc_layer = CC_module(out_channels)

                self.reasoning_convs.append(conv_layer)
                self.reasoning_ccs.append(cc_layer)


    # 根据通过tokenizer对输入文本进行编码，得到文档级表示和注意力信息
    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert" or config.transformer_type == 'deberta':
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        elif config.transformer_type == 'longformer':
            return process_long_input_longformer(self.model, input_ids, attention_mask)
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens, 512)
        return sequence_output, attention

    # 根据encode之后的文档表示和注意力信息，提取实体对嵌入，并通过图卷积增强这些嵌入
    def get_hrt(self, sequence_output, attention, hts, sent_pos, entity_pos, coref_pos, mention_pos, men_graphs, ne):
        offset = 1 if self.config.transformer_type in ["bert", "roberta", "longformer", "deberta"] else 0
        batch_size, num_neads, seq_len, seq_len = attention.size()
        batch_size, seq_len, hidden_size = sequence_output.size()
        hss, rss, tss = [], [], []
        device = sequence_output.device
        n_e = ne #最大实体数

        feats = []
        nms, nss, nes = [len(m) for m in mention_pos], [len(s) for s in sent_pos], [len(e) for e in entity_pos]
        for i in range(batch_size):
            doc_emb = sequence_output[i][0].unsqueeze(0)

            mention_embs = sequence_output[i, mention_pos[i] + offset]

            sentence_embs = [torch.logsumexp(sequence_output[i, offset + sent_pos[0]:offset + sent_pos[1]], dim=0) for
                             sent_pos in sent_pos[i]]
            sentence_embs = torch.stack(sentence_embs)

            all_embs = torch.cat([doc_emb, mention_embs, sentence_embs], dim=0)
            feats.append(all_embs)
        feats = torch.cat(feats, dim=0)
        assert len(feats) == batch_size + sum(nms) + sum(nss)
        feats = self.graph_conv(men_graphs, feats)

        cur_idx = 0
        batch_entity_embs, batch_entity_atts = [], []
        for i in range(batch_size):  # 
            entity_embs, entity_atts = [], []

            men_idx = -1
            for e_id, e in enumerate(entity_pos[i]):  # 
                if len(e) > 1:  # 
                    e_emb, g_emb, e_att = [], [], []
                    for start, end in e:  # 
                        men_idx += 1
                        if start + offset < seq_len:
                            e_emb.append(sequence_output[i, start + offset])
                            g_emb.append(feats[cur_idx + 1 + men_idx])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        if self.residual:
                            e_emb = torch.stack(e_emb) + torch.stack(g_emb)
                        else:
                            e_emb = torch.stack(g_emb)

                        # 新增：当coref为'none'时，直接聚合特征，不处理共指
                        if self.coref == 'none':
                            # 直接对多个提及的特征做logsumexp聚合
                            e_emb = torch.logsumexp(e_emb, dim=0)
                            # 直接对多个提及的注意力做平均
                            e_att = torch.stack(e_att).mean(0)

                        # 保留原有'gated'逻辑
                        elif self.coref == 'gated':
                            att = torch.stack(e_att).mean(0).sum(0)
                            gate_score = att / att.sum()
                            coref_emb = []
                            for start, end in coref_pos[i][e_id]:
                                coref_emb.append(
                                    (gate_score[start:end].unsqueeze(-1) * sequence_output[i, start:end]).sum(0))
                            if coref_emb:
                                e_emb = torch.cat([e_emb, torch.stack(coref_emb)])
                            e_emb = torch.logsumexp(e_emb, dim=0)
                            e_att = torch.stack(e_att).mean(0)

                        # 保留原有'e_context'逻辑
                        elif self.coref == 'e_context':
                            e_emb = torch.logsumexp(e_emb, dim=0)
                            for start, end in coref_pos[i][e_id]:
                                e_att.append(attention[i, :, start:end].mean(1))
                            e_att = torch.stack(e_att).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(num_neads, seq_len).to(attention)
                else:
                    start, end = e[0]
                    men_idx += 1
                    if start + offset < seq_len:
                        if self.residual:
                            e_emb = sequence_output[i, start + offset] + feats[cur_idx + 1 + men_idx]
                        else:
                            e_emb = feats[cur_idx + 1 + men_idx]
                        if self.coref == 'gated':
                            e_att = attention[i, :, start + offset]
                            att = e_att.sum(0)
                            gate_score = att / att.sum()
                            coref_emb = []
                            for start, end in coref_pos[i][e_id]:
                                coref_emb.append((gate_score[start:end].unsqueeze(-1) * sequence_output[i, start:end]).sum(0))
                            if coref_emb:
                                e_emb = torch.cat([e_emb.unsqueeze(0), torch.stack(coref_emb)])
                                e_emb = torch.logsumexp(e_emb, dim=0)
                        else:  # coref == 'e_context'
                            if not coref_pos[i][e_id]:
                                e_att = attention[i, :, start + offset]
                            else:
                                e_att = [attention[i, :, start + offset]]
                                for start, end in coref_pos[i][e_id]:
                                    e_att.append(attention[i, :, start:end].mean(1))
                                e_att = torch.stack(e_att).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(num_neads, seq_len).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            cur_idx += 1 + nms[i] + nss[i]
            entity_embs = torch.stack(entity_embs)
            entity_atts = torch.stack(entity_atts)
            batch_entity_embs.append(entity_embs)
            batch_entity_atts.append(entity_atts)

        # all_entity_embs = torch.cat(batch_entity_embs)
        cur_idx = 0
        for i in range(batch_size):
            # entity_embs = batch_entity_embs[i] + kg_feats[cur_idx:cur_idx + nes[i]]
            entity_embs = batch_entity_embs[i] #第一个batch中的实体数量
            cur_idx += nes[i]
            entity_atts = batch_entity_atts[i]
            s_ne, _ = entity_embs.size()

            ht_i = torch.LongTensor(hts[i]).to(device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            #按每个batch中最大的实体数
            # pad_hs = torch.zeros((n_e, hidden_size)).to(device)
            # pad_ts = torch.zeros((n_e, hidden_size)).to(device)
            # pad_hs[:s_ne, :] = entity_embs
            # pad_ts[:s_ne, :] = entity_embs

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            # ht_att = (h_att * t_att).mean(1)
            m = torch.nn.Threshold(0, 0)
            ht_att = m((h_att * t_att).sum(1))
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-10)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            # pad_rs = torch.zeros(n_e, hidden_size).to(device)
            # pad_rs[:s_ne, :] = rs

            #融合上下文感知信息
            hs_e = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
            ts_e = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
            #用融合后实体表征更新原始实体表示enetiy_embs
            new_entity_embs = torch.zeros_like(entity_embs)
            update_count = torch.zeros(s_ne).to(device) #[0, 0, 0, 0]
            #根据ht_i索引，将hs_e和ts_e更新new_entity_embs
            h_indices = ht_i[:, 0]
            t_indices = ht_i[:, 1]
            #更新头实体表示
            new_entity_embs.index_add_(0, h_indices, hs_e.float())
            update_count.index_add_(0, h_indices, torch.ones_like(h_indices).float()) #[1,1,1,0]
            #更新尾实体表示
            new_entity_embs.index_add_(0, t_indices, ts_e.float())
            update_count.index_add_(0, t_indices, torch.ones_like(t_indices).float()) #[1,3,1,1]
            #防止除零错误，得到融合上下文信息的实体表征
            update_count = update_count.clamp(min=1)
            new_entity_embs = new_entity_embs / update_count.unsqueeze(-1)

            # 按每个batch中最大的实体数
            pad_hs = torch.zeros((n_e, hidden_size)).to(device)
            pad_ts = torch.zeros((n_e, hidden_size)).to(device)
            pad_hs[:s_ne, :] = new_entity_embs
            pad_ts[:s_ne, :] = new_entity_embs

            hss.append(pad_hs)
            # rss.append(pad_rs)
            tss.append(pad_ts)
            # hss.append(hs)
            # rss.append(rs)
            # tss.append(ts)

        # hss = torch.cat(hss, dim=0) #[bs, num_hts, hidden_size]
        # tss = torch.cat(tss,dim=0)
        # rss = torch.cat(rss,dim=0)
        hss = torch.stack(hss)
        tss = torch.stack(tss)

        # return hss, rss, tss, batch_entity_embs
        return hss, tss

    def forward(self,
                input_ids,
                attention_mask,
                hts,
                sent_pos,
                entity_pos,
                coref_pos,
                mention_pos,
                entity_types,
                men_graphs,
                labels
                #   output_kg_scores=False
                ):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        batch_size, num_heads, seq_len, seq_len = attention.size()
        sequence_output[:, self.max_seq_length:, :] = 0
        device = sequence_output.device
        nes = [len(x) for x in entity_pos]
        ne = max(nes)
        nss = [len(x) for x in sent_pos]

        hs_e, ts_e = self.get_hrt(sequence_output, attention, hts, sent_pos, entity_pos, coref_pos, mention_pos, men_graphs, ne)
        #hs_e, ts_e [bs, ne, ne, emb]


        #添加TTM-RE
        if self.ttm_re:
            new_hs_e = []
            new_ts_e = []
            step = 256
            #遍历每个batch
            for batch_1 in range(0, hs_e.shape[0]):
                hs_e_e = hs_e[batch_1]
                ts_e_e = ts_e[batch_1]
                hs2 = []
                ts2 = []
                for batch_2 in range(0, hs_e.shape[1], step):
                    encoded = self.mu_encoder(
                        torch.cat([hs_e_e[batch_2:batch_2 + step].unsqueeze(1),
                                   ts_e_e[batch_2:batch_2 + step].unsqueeze(1)], dim=1).unsqueeze(1))
                    # print(encoded.shape, self.mu_encoder.memory_tokens.data.shape); quit()
                    hs2.append(encoded[:, 0, 0, :])
                    ts2.append(encoded[:, 0, 1, :])
                hs2 = torch.cat(hs2, dim=0)
                ts2 = torch.cat(ts2, dim=0)
                new_hs_e.append(hs2)
                new_ts_e.append(ts2)
            new_hs_e = torch.stack(new_hs_e)
            new_ts_e = torch.stack(new_ts_e)

            b1_e = (new_hs_e / 2 + hs_e / 2).view(batch_size, ne, self.emb_size // self.block_size, self.block_size)
            b2_e = (new_ts_e / 2 + ts_e / 2).view(batch_size, ne, self.emb_size // self.block_size, self.block_size)
            bl_e = (b1_e.unsqueeze(2) * b2_e.unsqueeze(1)).view(batch_size, ne, ne, -1)
        else:
            b1_e = hs_e.view(batch_size, ne, self.emb_size // self.block_size, self.block_size)
            b2_e = ts_e.view(batch_size, ne, self.emb_size // self.block_size, self.block_size)
            bl_e = (b1_e.unsqueeze(2) * b2_e.unsqueeze(1)).view(batch_size, ne, ne, -1)
        #十字交叉逻辑推理模块
        # elif self.reason_type == 'criss-cross':
        #     feature = self.projection(bl_e).permute(0, 3, 1, 2) #(bs, emb_size, num_e, num_e)
        #     r_rep_e = self.conv_reason_e_l1(feature)  # [batch_size, inter_channel, ent_num, ent_num]
        #     cc_output = self.cc_module(r_rep_e)
        #     #r_rep_e_2 = self.conv_reason_e_l2(cc_output)
        #     r_rep_e_2 = self.conv_reason_e_l2(cc_output).permute(0, 2, 3, 1)
        #     #cc_output_2 = self.cc_module(r_rep_e_2)
        #     #r_rep_e_3 = self.conv_reason_e_l3(cc_output_2).permute(0, 2, 3, 1)
        #     # r_rep_e_3 = self.conv_reason_e_l2(cc_output_2)
        #     # cc_output_3 = self.cc_module(r_rep_e_3)
        #     # r_rep_e_4 = self.conv_reason_e_l3(cc_output_3).permute(0, 2, 3, 1)
        #     #rel_logits = self.classifier(feature.permute(0, 2, 3, 1))
        #     rel_logits = self.classifier(r_rep_e_2)
        #     self_mask = (1 - torch.diag(torch.ones(ne))).unsqueeze(0).unsqueeze(-1).to(sequence_output)
        #     rel_logits = rel_logits * self_mask
        if self.reason_type == 'criss-cross':
            # 1. 初始变换：调整为 (Batch, Channel, Height, Width)
            current_features = self.projection(bl_e).permute(0, 3, 1, 2)

            # 2. 循环通过 L 层
            # 每一层都是：Conv -> CC_Attention
            for i in range(self.num_reasoning_layers):
                # 步骤 A: 卷积 (Conv + BN + ReLU)
                conv_out = self.reasoning_convs[i](current_features)

                # 步骤 B: Criss-Cross Attention
                # 注意：cc_output 将作为下一层的输入 (current_features)
                current_features = self.reasoning_ccs[i](conv_out)

            # 3. 循环结束后，current_features 的维度已经是 (B, 768, H, W)
            # 因为我们在 __init__ 里保证了最后一层的 out_channels 是 emb_size

            # 4. 调整维度给分类器 (B, H, W, 768)
            rel_logits = self.classifier(current_features.permute(0, 2, 3, 1))

            self_mask = (1 - torch.diag(torch.ones(ne))).unsqueeze(0).unsqueeze(-1).to(sequence_output)
            rel_logits = rel_logits * self_mask
            #取头尾实体索引
            flat_rel_logits = []
            for x in range(batch_size):
                hts_x = torch.tensor(hts[x]).to(device)
                selected_logits = rel_logits[x, hts_x[:, 0], hts_x[:, 1], :].reshape(-1, self.num_class)
                flat_rel_logits.append(selected_logits)
            flat_rel_logits = torch.cat(flat_rel_logits, dim=0)



        if labels is None:  #
            logits = flat_rel_logits
            return self.rel_loss_fnt.get_label(logits)
        else:  #
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(flat_rel_logits)
            output = self.rel_loss_fnt
            rel_loss = self.rel_loss_fnt(flat_rel_logits, labels)
            return rel_loss