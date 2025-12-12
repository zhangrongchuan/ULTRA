import os
from typing import List, Optional

import torch
from torch import Tensor

from sentence_transformers import SentenceTransformer

from relbench.datasets import get_dataset
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import make_pkey_fkey_graph
from torch_frame.config.text_embedder import TextEmbedderConfig

#导入数据并且拿到对应的图数据
# ==== 1. Load data ====
dataset = get_dataset("rel-f1", download=True)
db = dataset.get_db()

# ==== 2. get all the table and table info====
col_to_stype_dict = get_stype_proposal(db)

# ==== 3.define a text embedding model====
class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        # return shape [num_sent, emb_dim]--torch.Tensor
        return torch.from_numpy(self.model.encode(sentences))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_embedder_cfg = TextEmbedderConfig(
    text_embedder=GloveTextEmbedding(device=device),
    batch_size=256,
)

# ==== 4. construct pk to fk graph====
root_dir = "./relbench_cache"
os.makedirs(root_dir, exist_ok=True)

data, col_stats_dict = make_pkey_fkey_graph(
    db,
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=text_embedder_cfg,
    cache_dir=os.path.join(root_dir, "rel-f1_materialized_cache"),
)

# ==== 🚀 删除所有 reverse 关系 ====
reverse_edge_types = [
    etype for etype in data.edge_types
    if etype[1].startswith("rev_")
]

for etype in reverse_edge_types:
    # print("Removing reverse relation:", etype)
    del data[etype]

print("\nRemoved:", len(reverse_edge_types), "reverse relations\n")

# ==== 5. Print graph info ====
print("\n=== Graph info ===")
print("node type：", data.node_types)
print(len(data.node_types))
print("edge type：", data.edge_types)
print(len(data.edge_types))

for ntype in data.node_types:
    print(f"{ntype}: num_nodes = {data[ntype].num_nodes}")

for etype in data.edge_types:
    ei = data[etype].edge_index
    print(f"{etype}: edge_index shape = {ei.shape}")
    
    
###生成三元组列表 (s, r, o)###
# ====== 6. 给每个 node 分配全局实体 ID ======
node_offsets = {}
current = 0
for ntype in data.node_types:
    num = data[ntype].num_nodes
    node_offsets[ntype] = current
    current += num

print("node_offsets:", node_offsets)

# ====== 7. 给每一种 edge type 分配关系 ID ======
rel2id = {}          # (src_type, rel_name, dst_type) -> rel_id
relid2info = {}      # rel_id -> {src_type, rel_name, dst_type}
next_rel_id = 0
all_quads = []       # 存所有 (s, r, o)

for etype in data.edge_types:
    src_type, rel_name, dst_type = etype

    # 保证每一种 (src, rel, dst) 对应唯一的关系 ID
    if etype not in rel2id:
        rel2id[etype] = next_rel_id
        relid2info[next_rel_id] = {
            "src_type": src_type,
            "rel_name": rel_name,
            "dst_type": dst_type,
        }
        next_rel_id += 1

    r_id = rel2id[etype]

    # 当前 edge type 的所有边
    edge_index = data[etype].edge_index  # [2, num_edges]
    src_local = edge_index[0]            # [num_edges]
    dst_local = edge_index[1]

    # 转成全局实体 ID
    src_global = src_local + node_offsets[src_type]
    dst_global = dst_local + node_offsets[dst_type]

    # 生成 (s, r, o) triple
    triples_for_etype = torch.stack(
        [src_global, torch.full_like(src_global, r_id), dst_global],
        dim=-1  # [num_edges, 3]
    )
    all_quads.append(triples_for_etype)

# 拼成一个大的 [E_total, 3] tensor
all_quads = torch.cat(all_quads, dim=0)

print("\n=== Triple list (sample) ===")
print(all_quads[:50])
print("Total number of edges :", all_quads.shape[0])
print("Total num of relations:", len(rel2id))

from saved2local import *
# 保存entity映射到本地
save_entity_mapping(db, data, node_offsets, "./entities.tsv")
# 保存关系映射到本地
save_relation_mapping(relid2info, "./relations.tsv")
# 保存三元组到本地
save_all_triples(all_quads, path="./graph.txt")

if False:##if link prediction task, then do the following data split
    ###数据分割和过滤
    # all_quads: [E_total, 3]  (s, r, o)
    triples = all_quads.clone()

    # 1. 打乱顺序
    perm = torch.randperm(triples.shape[0])
    triples = triples[perm]

    # 2. 粗略按比例切 80% / 10% / 10%
    n_total = triples.shape[0]
    n_train = int(n_total * 0.8)
    n_valid = int(n_total * 0.1)

    train = triples[:n_train]
    valid = triples[n_train:n_train + n_valid]
    test  = triples[n_train + n_valid:]


    def get_entities_and_relations(triples_tensor: torch.Tensor):
        """从 [N,3] 的 (s,r,o) tensor 里提取实体集合和关系集合"""
        s = triples_tensor[:, 0]
        r = triples_tensor[:, 1]
        o = triples_tensor[:, 2]
        entities = set(s.tolist()) | set(o.tolist())
        relations = set(r.tolist())
        return entities, relations


    # 3. 确保 valid/test 中的 实体 / 关系类型 都在 train 里出现
    train_ents, train_rels = get_entities_and_relations(train)

    def filter_split(split, train_ents, train_rels):
        """把 split 里不满足(实体/关系已在 train 出现) 的样本移回 train"""
        keep = []
        move_to_train = []
        for t in split:
            s, r, o = t.tolist()
            if (s in train_ents) and (o in train_ents) and (r in train_rels):
                keep.append(t)
            else:
                move_to_train.append(t)
        if keep:
            keep = torch.stack(keep)
        else:
            keep = torch.empty((0, 3), dtype=split.dtype)
        if move_to_train:
            move_to_train = torch.stack(move_to_train)
        else:
            move_to_train = torch.empty((0, 3), dtype=split.dtype)
        return keep, move_to_train

    # 先处理 valid
    valid, back_to_train = filter_split(valid, train_ents, train_rels)
    if back_to_train.shape[0] > 0:
        train = torch.cat([train, back_to_train], dim=0)
        train_ents, train_rels = get_entities_and_relations(train)

    # 再处理 test
    test, back_to_train = filter_split(test, train_ents, train_rels)
    if back_to_train.shape[0] > 0:
        train = torch.cat([train, back_to_train], dim=0)
        train_ents, train_rels = get_entities_and_relations(train)

    print("Final sizes:")
    print("train:", train.shape[0])
    print("valid:", valid.shape[0])
    print("test :", test.shape[0])
    
    def save_triples(tensor, path):
        with open(path, "w") as f:
            for s, r, o in tensor.tolist():
                f.write(f"{s}\t{r}\t{o}\n")

    save_triples(train, "train.txt")
    save_triples(valid, "valid.txt")
    save_triples(test,  "test.txt")
    print("Saved train/valid/test triples.")
    
