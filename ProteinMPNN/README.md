# 给蛋白质结构计算ProteinMPNN 的embedding

修改cal_embedding.slurm中的变量
```sh
output_dir="./embedding_outputs" # s输出文件夹路径
seed=37 # 1种子
pdb_dir='~/embedding/alphafold_pdb' # 蛋白质结构文件夹，会计算此文件夹下面的所有结构的embeddng
```

