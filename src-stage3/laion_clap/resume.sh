#!/bin/bash
#SBATCH --comment clap
#SBATCH --partition=gamma
#SBATCH --job-name=mclap
#SBATCH --nodes 3
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-gpu=6
#SBATCH --exclusive
#SBATCH --output=%x_%j.out

module load cuda/11.3.1
module load mpi
export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export NCCL_DEBUG=info
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0
export WORLD_SIZE=8
# export MASTER_PORT=12802


OMP_NUM_THREADS=20 CUDA_LAUNCH_BLOCKING=1 torchrun --nnodes=1 --nproc_per_node 8 --master_addr=localhost --master_port=2120 -m training.main \
    --name "test-newfile-resume-2" \
    --save-frequency 1 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="csv" \
    --precision="fp32" \
    --batch-size=4 \
    --lr=1e-4 \
    --wd=0.0 \
    --epochs=50 \
    --workers=2 \
    --use-bn-sync \
    --amodel HTSAT-tiny \
    --tmodel t5 \
    --warmup 3200 \
    --report-to "wandb" \
    --wandb-notes "new_files_t5" \
    --train-data ./final_new.csv \
    --val-data ./final_val.csv \
    --top-k-checkpoint-select-dataset="Clotho-test" \
    --top-k-checkpoint-select-metric="mAP@10" \
    --openai-model-cache-dir ./laion_clap \
    --logs /fsx/clap_logs \
    --seed 3407 \
    --gather-with-grad \
    --optimizer "adam" \
    --data-filling "repeatpad" \
    --data-truncating "fusion" \
    --enable-fusion \
    --fusion-type "aff_2d" \
    --pretrained-audio ./htsat.ckpt \
    --resume 'path_to_resume_ckpt'