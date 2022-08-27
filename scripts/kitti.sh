python demo_optimization.py \
            --dataset kitti \
            --start_id 0 \
            --use_flow_consistency \
            --height 256 \
            --width 832 \
            --stage1_iters 3000 \
            --stage2_iters 10000 \
            --lr_stage1 0.1 \
            --lr_stage2 0.03 \
            --consist_weight 0.1 \
            --initial warp \
            --flow_ckpt kitti \
            --pred_steps 5 \
            --first_frame 2 \
            --second_frame 3 \
            --use_warp_im2 \
            --use_interp_im2 \
            --warp_weight 50 \
            --interp_weight 50 \
            --output_dir ./results/
