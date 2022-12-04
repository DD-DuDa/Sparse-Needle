import torch
import needle as ndl
from apex.contrib.sparsity import ASP

import matplotlib.pyplot as plt

# def torchNet(dim, hidden_dim=128, num_classes=10):
#     net = torch.nn.Sequential(
#         torch.nn.Linear(in_features=dim, out_features=hidden_dim), 
#         torch.nn.ReLU(), 
#         torch.nn.Linear(in_features=hidden_dim, out_features=10)
#     )

#     return net

# # TODO not abstract / only support cuda
# def model_to_sparse(ndl_model, device=ndl.cuda()):
#     torNet = torchNet(784, hidden_dim=128, num_classes=10)

#     """"""
#     new_Wight = torch.Tensor(ndl_model.parameters()[0].transpose().numpy())
#     torNet[0].weight = torch.nn.Parameter(new_Wight)

#     new_Wight = torch.Tensor(ndl_model.parameters()[2].transpose().numpy())
#     torNet[2].weight = torch.nn.Parameter(new_Wight)
#     """"""

#     ASP.init_model_for_pruning(torNet.cuda(), "m4n2_1d", allow_recompute_mask=True, verbosity=3, allow_permutation=False)
#     ASP.compute_sparse_masks()

#     """"""
#     ndl_model.parameters()[0].data = ndl.Tensor(torNet.state_dict()['0.weight'].t().cpu(), device=ndl.cuda())
#     ndl_model.parameters()[2].data = ndl.Tensor(torNet.state_dict()['2.weight'].t().cpu(), device=ndl.cuda())
#     """"""

def torchNet(dim, hidden_dim=128, num_classes=10):
    net = torch.nn.Sequential(
        torch.nn.Linear(in_features=hidden_dim, out_features=dim), 
        torch.nn.ReLU(), 
        torch.nn.Linear(in_features=dim, out_features=10)
    )

    return net

# TODO not abstract / only support cuda
def model_to_sparse(ndl_model, device=ndl.cuda()):
    torNet = torchNet(784, hidden_dim=128, num_classes=10)

    """"""
    new_Wight = torch.Tensor(ndl_model.parameters()[0].numpy())
    torNet[0].weight = torch.nn.Parameter(new_Wight)

    # new_Wight = torch.Tensor(ndl_model.parameters()[2].transpose().numpy())
    # torNet[2].weight = torch.nn.Parameter(new_Wight)
    """"""

    ASP.init_model_for_pruning(torNet.cuda(), "m4n2_1d", allow_recompute_mask=True, verbosity=3, allow_permutation=False)
    ASP.compute_sparse_masks()

    """"""
    ndl_model.parameters()[0].data = ndl.Tensor(torNet.state_dict()['0.weight'].cpu(), device=ndl.cuda())
    # ndl_model.parameters()[2].data = ndl.Tensor(torNet.state_dict()['2.weight'].t().cpu(), device=ndl.cuda())
    """"""



