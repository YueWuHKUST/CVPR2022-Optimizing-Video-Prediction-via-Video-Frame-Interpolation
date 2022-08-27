import numpy as np 
import torch

def fwd2bwd(fwd_flow, fwd_flow_conf):
    # manullay trun fwd flow to bwd flow
    # size if 1 x 2 x h x w
    _, _, h, w = fwd_flow.shape
    bwd_flow = np.zeros((1, 2, h, w))
    flags = np.zeros((1,2,h,w))
    from scipy import interpolate
    x_coordinates = []
    y_coordinates = []
    flow_x_values = []
    flow_y_values = []
    # store cooridnates for 2d interpolation
    for i in range(h):
        for j in range(w):
            #if fwd_flow_conf[0,0,i,j] > 0.9:
                # high confidence
            shift_x = fwd_flow[0,0,i,j]
            shift_y = fwd_flow[0,1,i,j]
            target_x = j + shift_x
            target_y = np.round(i + shift_y)
            # boundary dealing
            target_x = np.around(np.maximum(np.minimum(target_x, w - 1), 0))
            target_y = np.around(np.maximum(np.minimum(target_y, h - 1), 0))
            target_x = int(target_x)
            target_y = int(target_y)
            #print(-shift_x)
            #print(target_y, target_x)
            bwd_flow[0, 0, target_y, target_x] += -shift_x
            bwd_flow[0, 1, target_y, target_x] += -shift_y
            flags[0, 0, target_y, target_x] += 1
            flags[0, 1, target_y, target_x] += 1
            # avg among values
            if flags[0, 0, target_y, target_x] > 1:
                bwd_flow[0, 0, target_y, target_x] /= flags[0, 0, target_y, target_x]
                flags[0, 0, target_y, target_x] = 1
            if flags[0, 1, target_y, target_x] > 1:
                bwd_flow[0, 1, target_y, target_x] /= flags[0, 1, target_y, target_x]
                flags[0, 1, target_y, target_x] = 1

            y_coordinates.append(target_y)
            x_coordinates.append(target_x)
            flow_x_values.append(-shift_x)
            flow_y_values.append(-shift_y)
    
    # interpolate the zeros values
    # print("len x coordinates", len(x_coordinates))
    # print("")
    # error message: Too many data points to interpolate
    # func_x = interpolate.interp2d(x_coordinates, y_coordinates, flow_x_values, kind='linear')
    # func_y = interpolate.interp2d(x_coordinates, y_coordinates, flow_y_values, kind='linear')
    # # filling the holes
    # # target image coordinate system
    # for i in range(h):
    #     for j in range(w):
    #         if flags[0, 0, i, j] == 0:
    #             bwd_flow[0, 0, i, j] = func_x(j, i)
    #         if flags[0, 1, i, j] == 0:
    #             bwd_flow[0, 1, i, j] = func_y(j, i)
    return bwd_flow
    

# debug
# fwd_flow = np.ones((1,2,10,10))
# bwd_flow = fwd2bwd(fwd_flow)
# print(bwd_flow)