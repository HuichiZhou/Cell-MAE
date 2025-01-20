torchrun --nproc_per_node=1 train.py \
    --batch_size 32 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 100 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /home/zhhc/mae/train.csv


    