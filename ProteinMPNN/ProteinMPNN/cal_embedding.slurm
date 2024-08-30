#!/bin/bash
#SBATCH -J MPNN
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=20
#SBATCH -p fat
#SBATCH -w comput8
#SBATCH --mem=20G
#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH -o log/output.log
#SBATCH -e log/error.log
#SBATCH --mail-type=FAIL # BEGIN,END,FAIL,ALL
#SBATCH --mail-user=925201392@qq.com

output_dir="./outputs/embedding_outputs"
seed=37
xargs -a pdb.list -P 10 -I {} python /public/home/sunwk/embedding/ProteinMPNN/ProteinMPNN/protein_mpnn_run.py  --pdb_path {} --out_folder $output_dir --embedding 1 --sampling_temp "0.1"  --seed $seed --device cpu

