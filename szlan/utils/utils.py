import torch


def _ffd(forward_values, current_values, directions, h, nrm_const = 1.0):
#    print("size(curr): ", current_values.size(0))
    if current_values.ndim == 1: #and current_values.size() == 1:
        current_values = current_values[:, None]
    diff = (forward_values - current_values) /  h
#    print("DIFF", diff.shape, directions.shape)
    grad_contrib = diff[...,None] * directions#.view(-1, directions.shape[-1])
#    print("GRAD CONT: ", grad_contrib.shape)
#    return nrm_const * torch.mean(grad_contrib, dim=0)# / (b * l)
    
#    print(grad_contrib.shape, directions.shape)          
    b, l = directions.shape[:2]
    
#    print("B, L", b,l)

    return nrm_const * torch.sum(grad_contrib, dim=(0, 1)) / (b * l)




def _cfd(forward_values, backward_values, directions, h, nrm_const = 1.0):
    
    # directions is assumed to be (b, l, d)
    
    forward_values, backward_values = forward_values.view(-1, 1), backward_values.view(-1, 1)
    diff = (forward_values - backward_values) / (2 * h)
#    if directions.shape[0] == 1:
    grad_app = diff * directions
    # else:
    #     grad_app = diff[..., None] * directions
        
    b, l = directions.shape[:2]
    
    return nrm_const * torch.sum(grad_app, dim=(0, 1)) / (b * l)
    