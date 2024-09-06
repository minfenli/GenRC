import torch


def scale_shift_linear(rendered_depth, predicted_depth, mask, fuse=True):
    """
    Optimize a scale and shift parameter in the least squares sense, such that rendered_depth and predicted_depth match.
    Formally, solves the following objective:

    min     || (d * a + b) - d_hat ||
    a, b

    where d = 1 / predicted_depth, d_hat = 1 / rendered_depth

    :param rendered_depth: torch.Tensor (H, W)
    :param predicted_depth:  torch.Tensor (H, W)
    :param mask: torch.Tensor (H, W) - 1: valid points of rendered_depth, 0: invalid points of rendered_depth (ignore)
    :param fuse: whether to fuse shifted/scaled predicted_depth with the rendered_depth
                 if True, use rendered_depth when mask is 1 and use predicted_depth when mask is 0
                 if False, use predicted_depth anyway

    :return: scale/shift corrected depth
    """
    if mask.sum() == 0:
        return predicted_depth

    '''
    # shift and linear

    rendered_disparity = 1 / rendered_depth[mask].unsqueeze(-1)
    predicted_disparity = 1 / predicted_depth[mask].unsqueeze(-1)

    X = torch.cat([predicted_disparity, torch.ones_like(predicted_disparity)], dim=1)
    XTX_inv = (X.T @ X).inverse()
    XTY = X.T @ rendered_disparity
    AB = XTX_inv @ XTY

    fixed_disparity = (1 / predicted_depth) * AB[0] + AB[1]
    fixed_depth = 1 / fixed_disparity
    '''
    

    #'''
    # linear, no shift
    rendered_disparity = 1/rendered_depth[mask].unsqueeze(-1)
    predicted_disparity = 1/predicted_depth[mask].unsqueeze(-1)
    scale = (rendered_disparity.T @ predicted_disparity) / (predicted_disparity.T @ predicted_disparity)
    fixed_depth = predicted_depth / scale
    #'''

    if fuse:
        fused_depth = torch.where(mask, rendered_depth, fixed_depth)
        return fused_depth
    else:
        return fixed_depth
