import torch


def anomaly_score(input, x0, thresold_valid, config):
    ro = config.hypeparameter.beta * thresold_valid
    matrix_error = torch.square(torch.sub(input, x0))
    number_broken = len(matrix_error[matrix_error > ro])
    return number_broken

