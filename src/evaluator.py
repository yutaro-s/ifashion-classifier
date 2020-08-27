import time
from pathlib import Path
import numpy as np
import torch

from .metric import cal_score


def evaluator(args, model, data_loader, loss_fn):
    start_time = time.time()
    model.eval()
    loss_total = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device=args.device)
            y = y.to(device=args.device)

            pred = model(x)
            loss = loss_fn(pred, y)
            loss_total += loss.item() * x.size(0)

            y_true.append(y.data.cpu().numpy())
            pred.sigmoid_()
            y_pred.append(pred.data.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    score = cal_score(y_true, y_pred, threshold=0.5)

    # average
    loss_total = loss_total / data_loader.dataset.__len__()

    score['time'] = round(time.time() - start_time)
    score['loss'] = loss_total
    return score
