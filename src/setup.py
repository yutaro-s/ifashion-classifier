from logzero import setup_logger
#from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from adabound import AdaBound

from .model import ImgClassifier as Classifier
from .ifashion import iFashionAttribute as Dataset


def set_model(args, cfg, checkpoint):
    # model
    if checkpoint:
        model = Classifier(pretrained=False)
        model.load_state_dict(checkpoint['model'])
    else:
        model = Classifier(pretrained=True)
    if args.data_parallel:
        model = DataParallel(model)
    model = model.to(device=args.device)

    # optimizer
    if cfg['optimizer'] == 'sgd':
        optimizer = optim.ASGD(model.parameters(),
                               lr=cfg['learning_rate'],
                               weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg['learning_rate'],
                               weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'adabound':
        optimizer = AdaBound(model.parameters(),
                             lr=cfg['learning_rate'],
                             final_lr=0.1,
                             weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'amsbound':
        optimizer = AdaBound(model.parameters(),
                             lr=cfg['learning_rate'],
                             final_lr=0.1,
                             weight_decay=cfg['weight_decay'],
                             amsbound=True)

    # checkpoint
    if checkpoint and args.load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer


def set_loader(args, cfg):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.evaluation:
        train_loader = None
    else:
        train_loader = DataLoader(Dataset(
            root=args.train_dir,
            annFile=args.train_file,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
                                  batch_size=cfg['batch_size'],
                                  shuffle=args.loader_shuffle,
                                  drop_last=False,
                                  num_workers=args.num_workers,
                                  pin_memory=True)

    eval_loader = DataLoader(Dataset(root=args.eval_dir,
                                     annFile=args.eval_file,
                                     transform=transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize,
                                     ])),
                             batch_size=cfg['batch_size'],
                             shuffle=False,
                             drop_last=False,
                             num_workers=args.num_workers,
                             pin_memory=True)

    return train_loader, eval_loader


def set_logger(args, cfg):
    # logger
    if args.log:
        logger = setup_logger(name='logger',
                              logfile=args.output_dir + '/log.txt',
                              level=20,
                              disableStderrLogger=True)
    else:
        logger = setup_logger(name='logger',
                              logfile=None,
                              level=20,
                              disableStderrLogger=False)
    logger.debug(args)
    logger.debug(cfg)

    # tensorboard
    if args.tensorboard:
        pass
        #writer = SummaryWriter(log_dir=args.output_dir)
    else:
        writer = None
    return logger, writer
