from torch.nn import init, Module


def _init_weights(layer: Module, type: str) -> None:
    """Initialize layer weights

    Parameters
    ----------
    layer : Module
        Target layer.
    type : str
        Init method, can be ['Normal', 'Xavier', 'Kaiming', 'Orthogonal']

    Raises
    ------
    NotImplementedError
        Not supported init method.
    """
    classname = layer.__class__.__name__

    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        if type == "Normal":
            init.normal(layer.weight.data, 0.0, 0.02)
        elif type == "Xavier":
            init.xavier_normal(layer.weight.data, gain=0.02)
        elif type == "Kaiming":
            init.kaiming_normal(layer.weight.data, a=0, mode="fan_in")
        elif type == "Orthogonal":
            init.orthogonal(layer.weight.data, gain=1)
        else:
            raise NotImplementedError(
                f"Initialization method {type} is not implemented"
            )

    elif classname.find("BatchNorm2d") != -1:
        init.normal(layer.weight.data, 1.0, 0.02)
        init.constant(layer.bias.data, 0.0)


def init_weights(model: Module, type: str = "Normal"):
    model.apply(lambda x: _init_weights(x, type))
