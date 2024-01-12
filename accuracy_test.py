import time
import shutil
import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torchvision.datasets import ImageFolder

import models

class DefaultConfigs(object):
    # 1.string parameters
    train_dir = "/home/ubuntu/share/dataset/imagenet/train"
    val_dir = '/home/ubuntu/share/dataset/imagenet/val'
    model_name = "resnet18"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"

    # 2.numeric parameters
    epochs = 40
    start_epoch = 0
    batch_size = 256
    momentum = 0.9
    lr = 0.001
    weight_decay = 1e-4
    interval = 10
    workers = 5

    # 3.boolean parameters
    evaluate = False
    pretrained = False
    resume = False


device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0
#config = DefaultConfigs()


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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.001 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, weights, model_name, best_models):
    filename = weights + model_name + os.sep + "_checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best:
        message = best_models + model_name + os.sep + 'model_best.pth.tar'
        shutil.copyfile(filename, message)


def validate(val_loader, model, criterion, interval):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    # switch to evaluate mode
    model.eval()
    count = 0

    with torch.no_grad():
        end = time.time()
        for batch_id, (images, target) in enumerate(val_loader):
            images, target = images.cuda(), target.cuda()
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            count = count + 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (batch_id + 1) % interval == 0:
                progress.display(batch_id + 1)

        print(' * Acc@1 {top1.avg:.8f} Acc@5 {top5.avg:.8f}'
              .format(top1=top1, top5=top5))
        print("Count is " + str(count))
    return top1.avg


def train(train_loader, model, criterion, optimizer, epoch, interval):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()

    end = time.time()
    for batch_id, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images, target = images.cuda(), target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_id + 1) % interval == 0:
            progress.display(batch_id + 1)


def main():
    global best_acc
    epochs = 450
    start_epoch = 0
    batch_size = 64
    momentum = 0.9
    lr = 0.0001
    weight_decay = 1e-5
    interval = 10
    workers = 5
    weights = "/home/ml/Desktop/"
    best_models = weights + "best_model/"
    model_name = "moenas1"

    criterion = nn.CrossEntropyLoss().cuda()

    model = torch.load("./checkpoint/VGG/baseline/best.pth")
    #traindir = os.path.join("/all/uday/data/", 'train')
    model = nn.DataParallel(model)
    model.cuda()
    val_loader_path = '/home/ml3gpu/spar_pen/ucf/ucf50_dataset/validation'
    val_loader_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),transforms.Resize((224,224))])
    validation_dataset = ImageFolder(root=val_loader_path, transform=val_loader_transform)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=4)

    #valdir = os.path.join("/all/uday/data/", 'validation')
    
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    #optimizer = torch.optim.SGD(model.parameters(), lr,
                                #momentum=momentum,
                                #weight_decay=weight_decay)
    cudnn.benchmark = True

    ##if config.resume:
        #checkpoint = torch.load(config.best_models + "model_best.pth.tar")
        #config.start_epoch = checkpoint['epoch']
        #best_acc = checkpoint['best_acc']
        #model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])



    validate(val_loader, model, criterion, interval)
    # for batch_id, images in enumerate(val_loader):
    #     # images, target = images.cuda(), target.cuda()
    #     print(images)
            
    return


if __name__ == '__main__':
    main()
