import os
import argparse
from multiprocessing import Pool

def pool(idx):
    command = 'CUDA_VISIBLE_DEVICES=%d OMP_NUM_THREADS=2 python demo_optimization.py \
            --dataset cityscapes \
            --start_id %d \
            --length %d \
            --use_interp_im2 \
            --use_flow_consistency \
            --height %d \
            --width %d \
            --stage1_iters 3000 \
            --stage2_iters 10000 \
            --lr_stage1 0.1 \
            --lr_stage2 0.5 \
            --consist_weight 0.1' % (1, idx * 100 + 200 + 200, 100, 512, 1024)

    print(command)
    os.system(command)

if __name__ == '__main__':
    with Pool(8) as p:
        print(p.map(pool, range(8)))