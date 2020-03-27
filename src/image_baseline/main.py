import argparse
import os
import time
from collections import OrderedDict
import datasets
from callbacks import Logger
import torch
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from albumentations import Compose, ISONoise, JpegCompression, Downscale, Normalize, HorizontalFlip, Resize, RandomBrightnessContrast, RandomGamma, CLAHE
from albumentations.pytorch import ToTensor
from loss import FocalLoss

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--datadir', default='/bigssd/joanna/fakerecog/data/dfdc_bursted_final/', type=str, help='Location of the main json file')
parser.add_argument('--epochs', default=62, type=int, help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float, help='initial learning rate', dest='lr')
parser.add_argument('-p', '--print-freq', default=200, type=int, help='print frequency (default: 10)')
parser.add_argument('--logdir', type=str, default='../../logs/', help='folder to store log files')
parser.add_argument('--log_freq', '-l', default=500, type=int, help='frequency to write in tensorboard (default: 10)')
parser.add_argument('--exp', type=str, default='test', help='experiment name')
parser.add_argument('--temp', default=1, type=float, help='temperature')
parser.add_argument('--lossfunc', default='crossentropy', choices=['crossentropy','focal'], type=str, help='Loss function to train with')


def main():
    args = parser.parse_args()
    best_acc1 = 0

    # Initialize logger
    tb_logdir = os.path.join(args.logdir, args.exp)
    os.makedirs(tb_logdir, exist_ok=True)
    tb_logger = Logger(tb_logdir)
    print('==> Opts and logger initialized..')

    # Initialize datasets and transforms
    valfolders = [0]

    base_transforms = [Resize(224, 224), Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],), ToTensor()]
    #train_extra = [ISONoise(), RandomBrightnessContrast(), RandomGamma(), CLAHE(), HorizontalFlip(), JpegCompression(quality_lower=19, quality_upper=100, p=0.75), Downscale(scale_min=0.25, scale_max=0.99, p=0.5)]
    train_extra = [HorizontalFlip()]

    train_transforms = Compose(train_extra+base_transforms)
    val_transforms = Compose(base_transforms)
    train_dataset = datasets.SimpleFolderLoader(root=args.datadir, split='train', valfolders=valfolders, transform=train_transforms)
    val_dataset = datasets.SimpleFolderLoader(root=args.datadir, split='val', valfolders=valfolders, transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    print('==> Train and val loaders initialized..')

    # Initialize loss function
    weights = [.8, .2]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda()

    # Import a pretrained model
    model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=2)
    # Phase 1: Train the last layer
    # Set only last layer for learning
    for param in model.parameters():
        param.requires_grad = False
    model._fc.weight.requires_grad = True
    model._fc.bias.requires_grad = True
    model = model.cuda()

    # Initialize optimizer for training Phase 1
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-8)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=2e-6)
    torch.backends.cudnn.benchmark = True
    print('==> Model, optimizer, criterion initialized..')

    print('==> Training last layer..')
    # Train for one epoch and evaluate on validation set
    for epoch in range(6):
        train(loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch, iterations=750, args=args, tb_logger=tb_logger)
        lr_scheduler.step()
    acc1, nll = test(loader=val_loader, model=model, criterion=criterion, args=args, epoch=0, tb_logger=tb_logger)

    # Phase 2: Train all layers
    # Set all layers for learning, and parallelize over GPUs
    for param in model.parameters():
        param.requires_grad = True

    # Reset optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-8)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=1e-6)
    torch.backends.cudnn.benchmark = True

    for epoch in range(args.epochs):
        # train for one epoch and evaluate on validation set
        train(loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch+1, iterations=750, args=args, tb_logger=tb_logger)
        lr_scheduler.step()

    acc1, nll = test(loader=val_loader, model=model, criterion=criterion, args=args, epoch=epoch+1, tb_logger=tb_logger)
    filename = args.logdir +'/'+ args.exp + '/' + 'ckpt.pth.tar'
    torch.save({'acc1': acc1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()}, filename)


def train(loader, model, criterion, optimizer, epoch, args, iterations, tb_logger=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    print("==> Starting pass number: "+str(epoch)+", Learning rate: " + str(optimizer.param_groups[-1]['lr']))
    end = time.time()
    for i, (images, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #if args.gpu is not None:
        images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0][0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


        # log training data into tensorboard
        if tb_logger is not None and i % args.log_freq == 0:
            logs = OrderedDict()
            logs['Train_IterLoss'] = losses.val
            logs['Train_Acc@1'] = top1.val
            # how many iterations we have trained
            iter_count = epoch * iterations + i
            for key, value in logs.items():
                tb_logger.log_scalar(value, key, iter_count)

            tb_logger.flush()
        
        if i >= iterations:
            break 
        

def test(loader, model, criterion, args, epoch=None, tb_logger=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', '')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            # import pdb
            # pdb.set_trace()
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0][0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} '.format(top1=top1))
        print(' * Loss@1 {losses.avg:.4f} '.format(losses=losses))

    if epoch is not None and tb_logger is not None:
        logs = OrderedDict()
        logs['Val_EpochLoss'] = losses.avg
        logs['Val_EpochAcc@1'] = top1.avg
        # how many iterations we have trained
        for key, value in logs.items():
            tb_logger.log_scalar(value, key, epoch + 1)
        tb_logger.flush()
    return top1.avg, losses.avg    


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
