import torch

def cross_entropy_loss(inputs, targets):
    """
    inputs: (batch_size, num_classes), unnormalized logits
    targets: (batch_size,), correct class indices
    """

    x = inputs - torch.max(inputs, axis=-1, keepdims=True).values
    tmpx = torch.log(torch.sum(torch.exp(x), axis=-1))    # (batch_size,)

    output = x - tmpx.unsqueeze(-1) # (batch_size, num_classes)
    losses = -torch.gather(output, dim=1, index=targets.unsqueeze(-1)).squeeze(1)

    return torch.mean(losses)

