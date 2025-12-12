import csv

def save_entity_mapping(db, data, node_offsets, path: str = "entities.tsv"):
    """
    导出实体 ID 映射：
        global_id, node_type, primary_key, name
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["global_id", "node_type", "primary_key", "name"])

        for ntype in data.node_types:
            table = db.table_dict[ntype]  # 每个表
            df = table.df.reset_index(drop=True)
            pkey_col = table.pkey_col    # 主键列名（可能是 None）

            # 自动找 name 或 title 列（可选）
            name_col = None
            for col in df.columns:
                if "name" in col.lower() or "title" in col.lower():
                    name_col = col
                    break

            for local_id in range(len(df)):
                global_id = node_offsets[ntype] + local_id

                # 读取主键
                if pkey_col is not None:
                    pk_value = df.iloc[local_id][pkey_col]
                else:
                    pk_value = ""

                # 读取可读名称
                if name_col is not None:
                    name_value = df.iloc[local_id][name_col]
                else:
                    name_value = ""

                # ⚠️ 删除 local_id
                writer.writerow([
                    int(global_id),
                    ntype,
                    pk_value,
                    name_value,
                ])

    print(f"Saved entity mapping to {path}")
    
    
def save_relation_mapping(relid2info, path: str = "relations.tsv"):
    """
    导出关系 ID 映射：
        rel_id, src_type, rel_name, dst_type
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["rel_id", "src_type", "rel_name", "dst_type"])

        for rel_id, info in sorted(relid2info.items(), key=lambda x: x[0]):
            writer.writerow([
                int(rel_id),
                info["src_type"],
                info["rel_name"],
                info["dst_type"],
            ])

    print(f"Saved relation mapping to {path}")

def save_all_triples(all_quads, path="graph.txt"):
    """
    将所有 triples (s, r, o) 保存为 TSV 文件。
    每一行格式:  s<TAB>r<TAB>o
    """
    with open(path, "w", encoding="utf-8") as f:
        for s, r, o in all_quads.tolist():
            f.write(f"{s}\t{r}\t{o}\n")

    print(f"Saved {all_quads.shape[0]} triples to {path}")