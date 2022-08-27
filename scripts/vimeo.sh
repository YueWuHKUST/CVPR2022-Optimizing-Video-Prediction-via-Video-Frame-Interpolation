python demo_optimization.py \
            --dataset vimeo \
            --start_id 0 \
            --length 100 \
            --use_flow_consistency \
            --stage1_iters 3000 \
            --stage2_iters 10000 \
            --lr_stage1 0.1 \
            --lr_stage2 0.03 \
            --consist_weight 0.3 \
            --initial warp \
            --flow_ckpt kitti \
            --pred_steps 3 \
            --first_frame 0 \
            --second_frame 1 \
            --use_warp_im2 \
            --warp_weight 100 \
            --output_dir ./results/