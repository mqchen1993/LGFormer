CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 15360 train.py \
                                                        --swin_type base \
                                                        --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
                                                        --refer_data_root ./datasets/ \
                                                        --dataset refcocog \
                                                        --splitBy umd \
                                                        --img_size 480 \
                                                        --batch_size 16 \
                                                        --pin_mem \
                                                        --lr 5e-5 \
                                                        --wd 1e-2 \
                                                        --epochs 40 \
                                                        --amp \
                                                        2>&1 | tee ./models/refcoco/output