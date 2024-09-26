import torch


def convert_to_tensor(array):
    return torch.tensor(array)


def convert_to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy()
    else:
        return tensor
