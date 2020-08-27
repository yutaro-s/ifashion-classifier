import time
from pathlib import Path


def trainer(args, model, data_loader, loss_fn, optimizer):
    start_time = time.time()
    model.train()
    loss_total = 0.0

    for x, y in data_loader:
        x = x.to(device=args.device)
        y = y.to(device=args.device)
        optimizer.zero_grad()

        pred = model(x)
        loss = loss_fn(pred, y)
        loss_total += loss.item() * x.size(0)
        loss.backward()
        optimizer.step()

    # average
    loss_total = loss_total / data_loader.dataset.__len__()

    return {'time': round(time.time() - start_time), 'loss': loss_total}
