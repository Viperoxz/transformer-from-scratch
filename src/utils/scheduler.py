def rate(step, model_size, factor, warmup):
    """
    Implements the learning rate schedule from "Attention is All You Need".
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )
