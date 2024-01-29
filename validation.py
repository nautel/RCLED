####################



# this is test# this is test# this is test# this is test# this is test

import torch


def shrink(epsilon, input):
    x = input.copy()
    t1 = x > epsilon
    t2 = x < epsilon
    t3 = x > -epsilon
    t4 = x < -epsilon
    x[t2 & t3] = 0
    x[t1] = x[t1] - epsilon
    x[t4] = x[t4] + epsilon
    return x

def trans_one2five(x_in):
    x_1 = torch.roll(x_in, 1, dims=0)
    x_2 = torch.roll(x_in, 2, dims=0)
    x_3 = torch.roll(x_in, 3, dims=0)
    x_4 = torch.roll(x_in, 4, dims=0)
    return torch.cat((x_4, x_3, x_2, x_1, x_in), 0)

