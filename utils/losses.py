import torch

def l1_loss_TMN(V1, V2, average=True):
    """ Standard L1 loss
    Input : V1, V2 = Batchsize x N x Dimension
    average = True (mean) False(sum over all pts)
    Output : Loss = L1 loss
    """

    Loss = torch.abs(V1 - V2)
    Loss = torch.sum(Loss, 2)  # sum error in the last dimension

    if average:
        Loss = torch.mean(Loss, 1)
    else:
        Loss = torch.sum(Loss, 1)  # sum the error in all points

    # computing the mean across batches
    return Loss.mean()
