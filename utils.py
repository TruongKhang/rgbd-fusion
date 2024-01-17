import torch


def make_recursive_func(func):
    def wrapper(vars, device):
        if isinstance(vars, list):
            return [wrapper(x, device) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x, device) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v, device) for k, v in vars.items()}
        else:
            return func(vars, device)

    return wrapper


# load data into GPU device
@make_recursive_func
def to_device(vars, device):
    if isinstance(vars, torch.Tensor):
        return vars.to(device)
    elif isinstance(vars, str):
        return vars
    elif isinstance(vars, int):
        return vars
    else:
        raise NotImplementedError("invalid input type {}".format(type(vars)))