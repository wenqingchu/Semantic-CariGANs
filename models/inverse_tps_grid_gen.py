# encoding: utf-8

import torch
import itertools
import torch.nn as nn
from torch.autograd import Function, Variable

# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation, very slow
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist+0.0001)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix

def print_grad(grad):
    print('inverse_tps_grid_gen')
    print(grad.data.cpu())

class InverseTPSGridGen(nn.Module):

    def __init__(self):
        super(InverseTPSGridGen, self).__init__()



        # register precomputed matrices
        self.register_buffer('padding_matrix', torch.zeros(3, 2))

    def forward(self, target_height, target_width, source_control_points, target_control_points):
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3).cuda(target_control_points.get_device())
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)

        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))

        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        HW = target_height * target_width
        target_coordinate = list(itertools.product(range(target_height), range(target_width)))
        target_coordinate = torch.Tensor(target_coordinate) # HW x 2
        Y, X = target_coordinate.split(1, dim = 1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y], dim = 1) # convert from (y, x) to (x, y)
        target_coordinate = target_coordinate.cuda(target_control_points.get_device())
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate.cuda(target_control_points.get_device()), target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1).cuda(target_control_points.get_device()), target_coordinate
        ], dim = 1)



        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)

        Y = torch.cat([source_control_points, self.padding_matrix.expand(batch_size, 3, 2).cuda(source_control_points.get_device())], 1)


        mapping_matrix = torch.matmul(inverse_kernel, Y)

        source_coordinate = torch.matmul(target_coordinate_repr, mapping_matrix)
        return source_coordinate
