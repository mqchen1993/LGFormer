CUDA_VISIBLE_DEVICES=0 python test.py --swin_type base \
                                      --window12 \
                                      --refer_data_root ./datasets/ \
                                      --dataset refclef \
                                      --split val \
                                      --img_size 480 \
                                      --workers 4 \
                                      --ddp_trained_weights \
                                      --resume ./checkpoints/model_referit.pth