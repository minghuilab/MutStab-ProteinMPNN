#!/bin/bash
#SBATCH -J MPNN
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=40
#SBATCH -p fat
#SBATCH -w comput7
#SBATCH --mem=150G
#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH -o log/output.log
#SBATCH -e log/error.log
#SBATCH --mail-type=FAIL # BEGIN,END,FAIL,ALL
#SBATCH --mail-user=925201392@qq.com

output_dir="./embedding_outputs"
seed=37
pdb_dir='~/embedding/alphafold_pdb'
#cp ../ProteinMPNN/cal_embedding.slurm  .
#mkdir -p embedding_outputs
#mkdir -p log
if [ -d $output_dir"/embedding" ]; then
	echo 'continue calculate embeddihng '
	ls  $output_dir"/embedding" | sed 's/.npz//g' > caled
	find $pdb_dir | sed '1d' | grep -vf caled |  xargs -P 10 -I {} python ~/embedding/ProteinMPNN/ProteinMPNN/protein_mpnn_run.py  --pdb_path {} --out_folder $output_dir --embedding 1 --sampling_temp "0.1"  --seed $seed --device cpu
else
	find $pdb_dir | sed '1d'  | xargs -P 10 -I {} python ~/embedding/ProteinMPNN/ProteinMPNN/protein_mpnn_run.py  --pdb_path {} --out_folder $output_dir --embedding 1 --sampling_temp "0.1"  --seed $seed --device cpu
fi
