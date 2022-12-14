from torch.nn import init, Module


def _init_weights(layer: Module, method: str) -> None:
    """Initialize layer weights

    Parameters
    ----------
    layer : Module
        Target layer.
    method : str
        Init method, can be ['Normal', 'Xavier', 'Kaiming', 'Orthogonal']

    Raises
    ------
    NotImplementedError
        Not supported init method.
    """
    classname = layer.__class__.__name__

    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        if method == "Normal":
            init.normal_(layer.weight.data, 0.0, 0.02)
        elif method == "Xavier":
            init.xavier_normal_(layer.weight.data, gain=0.02)
        elif method == "Kaiming":
            init.kaiming_normal_(layer.weight.data, a=0, mode="fan_in")
        elif method == "Orthogonal":
            init.orthogonal_(layer.weight.data, gain=1)
        else:
            raise NotImplementedError(
                f"Initialization method {method} is not implemented"
            )

    elif classname.find("BatchNorm2d") != -1:
        init.normal_(layer.weight.data, 1.0, 0.02)
        init.constant_(layer.bias.data, 0.0)


def init_weights(model: Module, method: str = "Normal"):
    """Initialize model weights

    Parameters
    ----------
    model : Module
        Target model.
    method : str
        Init method, can be ['Normal', 'Xavier', 'Kaiming', 'Orthogonal']
    """
    model.apply(lambda x: _init_weights(x, method))
