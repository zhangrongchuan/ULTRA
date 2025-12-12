import numpy as np
import pandas as pd

from relbench.datasets import get_dataset
from relbench.tasks import get_task
import os
# ============================
# 0. 配置区域：根据自己情况修改
# ============================
def relbench_f1_top3_to_triples():

    # driver 节点在 entities_df 里对应的 node_type 名字
    NODE_TYPE_DRIVERS = "drivers"   # 如果你那一列不是 "drivers"，改这里

    # 这个 relation 的 id，以及两个 tail 的 id
    REL_ID = 13
    TAIL_ID_FOR_QUAL_0 = 74063
    TAIL_ID_FOR_QUAL_1 = 74064

    # ============================
    # 1. 加载 dataset 和 task
    # ============================

    dataset = get_dataset("rel-f1", download=True)
    task = get_task("rel-f1", "driver-top3", download=True)

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    print("Train head:")
    print(train_table.df.head())

    # ============================
    # 2. 加载你的实体列表 entities_df
    #    这个表应该有以下列：
    #    ['global_id', 'node_type', 'local_id', 'primary_key', 'name', 'new_global_id']
    # ============================

    # ⚠️ 这里替换成你自己的加载方式，比如：
    # entities_df = pd.read_parquet("entities.parquet")
    # 或者：
    # entities_df = pd.read_csv("entities.csv")

    # 下面这行只是示意，记得改掉！
    entities_df = pd.read_csv("./entities.tsv", sep="\t")

    print("Entities head:")
    print(entities_df.head())

    # ============================
    # 3. 建立 driverId -> new_global_id 映射
    # ============================

    drivers_map = (
        entities_df
        .query("node_type == @NODE_TYPE_DRIVERS")
        [["primary_key", "global_id"]]
        .rename(columns={"primary_key": "driverId"})
    )

    print("Drivers map head:")
    print(drivers_map.head())


    def make_triples_for_split(table, drivers_map,
                            rel_id=REL_ID,
                            tail0=TAIL_ID_FOR_QUAL_0,
                            tail1=TAIL_ID_FOR_QUAL_1):
        """
        从一个 split 的 table (train/val/test) 生成三元组：
            (driver_new_global_id, rel_id, tail_id)

        其中：
            - driver_new_global_id 通过 driverId -> new_global_id 映射得到
            - tail_id 由 qualifying == 0 / 1 决定
        """
        df = table.df.copy()

        # 合并 driverId -> new_global_id
        df = df.merge(drivers_map, on="driverId", how="left")

        # 检查是否有没匹配上的 driver
        missing_mask = df["global_id"].isna()
        if missing_mask.any():
            print("WARNING: some driverId not found in entities_df!")
            print(df.loc[missing_mask, ["driverId"]].drop_duplicates().head())
            # 如果你希望直接报错，可以改成：
            # raise ValueError("Some driverId not found in entities_df")

        # 填 relation_id
        df["relation_id"] = rel_id

        # qualifying == 0 -> tail0，qualifying == 1 -> tail1
        df["tail_id"] = np.where(df["qualifying"] == 0, tail0, tail1)

        # 只保留需要的列，构成 (head, rel, tail) 三元组
        triples = df[["global_id", "relation_id", "tail_id"]].to_numpy(dtype=np.int64)

        return triples, df


    # ============================
    # 4. 分别生成 train / val / test 的三元组
    # ============================

    train_triples, train_df_full = make_triples_for_split(train_table, drivers_map)
    val_triples,   val_df_full   = make_triples_for_split(val_table, drivers_map)
    # test_triples,  test_df_full  = make_triples_for_split(test_table, drivers_map)

    print("Train triples shape:", train_triples.shape)
    print("Val triples shape:", val_triples.shape)
    # print("Test triples shape:", test_triples.shape)

    # 看前几行三元组确认一下：
    print("Sample train triples (head, rel, tail):")
    print(train_triples[:10])

    # ============================
    # 5. 保存 train / val triples 为文本文件
    # ============================

    # 保存 train
    np.savetxt("train.txt", train_triples, fmt="%d")

    # 保存 val
    np.savetxt("valid.txt", val_triples, fmt="%d")

    print("train.txt and valid.txt have been saved!")
    
    
def relbench_f1_dnf_to_triples():

    # driver 节点在 entities_df 里对应的 node_type 名字
    NODE_TYPE_DRIVERS = "drivers"   # 如果你那一列不是 "drivers"，改这里

    # 这个 relation 的 id，以及两个 tail 的 id
    REL_ID = 13
    TAIL_ID_FOR_QUAL_0 = 74063
    TAIL_ID_FOR_QUAL_1 = 74064

    # ============================
    # 1. 加载 dataset 和 task
    # ============================

    dataset = get_dataset("rel-f1", download=True)
    task = get_task("rel-f1", "driver-dnf", download=True)

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    print("Train head:")
    print(train_table.df.head())

    # ============================
    # 2. 加载你的实体列表 entities_df
    #    这个表应该有以下列：
    #    ['global_id', 'node_type', 'local_id', 'primary_key', 'name', 'new_global_id']
    # ============================

    # ⚠️ 这里替换成你自己的加载方式，比如：
    # entities_df = pd.read_parquet("entities.parquet")
    # 或者：
    # entities_df = pd.read_csv("entities.csv")

    # 下面这行只是示意，记得改掉！
    entities_df = pd.read_csv("./entities.tsv", sep="\t")

    print("Entities head:")
    print(entities_df.head())

    # ============================
    # 3. 建立 driverId -> new_global_id 映射
    # ============================

    drivers_map = (
        entities_df
        .query("node_type == @NODE_TYPE_DRIVERS")
        [["primary_key", "global_id"]]
        .rename(columns={"primary_key": "driverId"})
    )

    print("Drivers map head:")
    print(drivers_map.head())


    def make_triples_for_split(table, drivers_map,
                            rel_id=REL_ID,
                            tail0=TAIL_ID_FOR_QUAL_0,
                            tail1=TAIL_ID_FOR_QUAL_1):
        """
        从一个 split 的 table (train/val/test) 生成三元组：
            (driver_new_global_id, rel_id, tail_id)

        其中：
            - driver_new_global_id 通过 driverId -> new_global_id 映射得到
            - tail_id 由 did_not_finish == 0 / 1 决定
        """
        df = table.df.copy()

        # 合并 driverId -> new_global_id
        df = df.merge(drivers_map, on="driverId", how="left")

        # 检查是否有没匹配上的 driver
        missing_mask = df["global_id"].isna()
        if missing_mask.any():
            print("WARNING: some driverId not found in entities_df!")
            print(df.loc[missing_mask, ["driverId"]].drop_duplicates().head())
            # 如果你希望直接报错，可以改成：
            # raise ValueError("Some driverId not found in entities_df")

        # 填 relation_id
        df["relation_id"] = rel_id

        # did_not_finish == 0 -> tail0，did_not_finish == 1 -> tail1
        df["tail_id"] = np.where(df["did_not_finish"] == 0, tail0, tail1)

        # 只保留需要的列，构成 (head, rel, tail) 三元组
        triples = df[["global_id", "relation_id", "tail_id"]].to_numpy(dtype=np.int64)

        return triples, df


    # ============================
    # 4. 分别生成 train / val / test 的三元组
    # ============================

    train_triples, train_df_full = make_triples_for_split(train_table, drivers_map)
    val_triples,   val_df_full   = make_triples_for_split(val_table, drivers_map)
    # test_triples,  test_df_full  = make_triples_for_split(test_table, drivers_map)

    print("Train triples shape:", train_triples.shape)
    print("Val triples shape:", val_triples.shape)
    # print("Test triples shape:", test_triples.shape)

    # 看前几行三元组确认一下：
    print("Sample train triples (head, rel, tail):")
    print(train_triples[:10])

    # ============================
    # 5. 保存 train / val triples 为文本文件
    # ============================

    # 保存 train
    np.savetxt("train.txt", train_triples, fmt="%d")

    # 保存 val
    np.savetxt("valid.txt", val_triples, fmt="%d")

    print("train.txt and valid.txt have been saved!")
    
# relbench_f1_top3_to_triples()
relbench_f1_dnf_to_triples()