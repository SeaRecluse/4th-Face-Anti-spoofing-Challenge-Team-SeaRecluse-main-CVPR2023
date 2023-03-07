import argparse
import csv
import os
import random
import sys
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel

import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import trange
import numpy as np
from clr import CyclicLR
from data import get_loaders
from logger import CsvLogger
from run import correct, save_checkpoint, find_bounds_clr
from network_util import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy,mixup_data
import timm
from tqdm import tqdm, trange

data_val_path = "./orig_data/data_div/val/"
fake_class_nums = 1
model_title ="convnext_base_in22ft1k"
epoch_scale = 3

parser = argparse.ArgumentParser(description='resnet18 training with PyTorch')
parser.add_argument('--dataroot', metavar='PATH', default='./orig_data/data_div/', 
                    help='Path to ImageNet train and val folders, preprocessed as described in '
                         'https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset')
parser.add_argument('--gpus', default="0", help='List of GPUs used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='Number of data loading workers (default: 4)')
parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64. Default: float32')

# data augment
parser.add_argument('--class_nums', type = int, default = 2, help='the nums of class.')
parser.add_argument('--img_size', type = int, default = 224, help='the size of image input to the model.')
parser.add_argument('--label_smooth', type = float, default = 0.1, help='the lam of label smoothing.')
parser.add_argument('--mixup_prob', type = float, default = 0.5, help='the prob of mixup.')
parser.add_argument('--mixup_alpha', type = float, default = 0.2, help='the alpha of mixup.')

# Optimization optionss
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('-b', '--batch-size', default = 64, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.005, help='The learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=4e-5, help='Weight decay (L2 penalty).')
parser.add_argument('--gamma', type=float, default=0.01, help='LR is multiplied by gamma at scheduled epochs.')
parser.add_argument('--schedule', type=int, nargs='+', default=[60 // epoch_scale, 120 // epoch_scale, 180 // epoch_scale, 240 // epoch_scale, 480 // epoch_scale, 960 // epoch_scale],
                    help='Decrease learning rate at these epochs.')

# CLR
parser.add_argument('--clr', dest='clr', default=True, help='Use CLR')
parser.add_argument('--min-lr', type=float, default=1e-5, help='Minimal LR for CLR.')
parser.add_argument('--max-lr', type=float, default=0.002, help='Maximal LR for CLR.')
parser.add_argument('--epochs-per-step', type=int, default=30,
                    help='Number of epochs per step in CLR, recommended to be between 2 and 10.')
parser.add_argument('--mode', default='exp_range', help='CLR mode. One of {triangular, triangular2, exp_range}')
parser.add_argument('--find-clr', dest='find_clr', action='store_true',
                    help='Run search for optimal LR in range (min_lr, max_lr)')

# Checkpointss
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='Just evaluate model')
parser.add_argument('--save', '-s', type=str, default='', help='Folder to save checkpoints.')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='Directory to store results')

parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

#parser.add_argument('--resume', default='./checkpoint.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
#parser.add_argument('--start-epoch', default=60 // epoch_scale, type=int, metavar='N', help='manual epoch number (useful on restarts)')

#parser.add_argument('--resume', default='./model_best.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
#parser.add_argument('--start-epoch', default=60 // epoch_scale, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='Number of batches between log messages')
parser.add_argument('--seed', type=int, default=3407, metavar='S', help='random seed (default: random)')

def train(model, loader, epoch, optimizer, criterion, device, dtype, batch_size, log_interval, scheduler, args):
    model.train()
    correct1, correct2 = 0, 0

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        #print(data.shape)
        if isinstance(scheduler, CyclicLR):
            scheduler.batch_step()

        data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        use_mixup = np.random.rand() < args.mixup_prob
        mixup_alpha_v = 0
        if use_mixup:
            mixup_alpha_v = args.mixup_alpha
        if args.label_smooth:
            data, target, y_a, y_b, lam = mixup_data(data, target,num_classes=2, alpha=mixup_alpha_v, label_smoothing = args.label_smooth, device=device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        if args.label_smooth > 0:
            _, y_pred = torch.max(output, 1)
            _,target = torch.max(target, 1)
      
        loss.backward()
        optimizer.step()
        if args.label_smooth > 0:
            step_corrects = y_pred.eq(target.data).cpu().sum().double()
            correct1 += step_corrects
            correct2=1
        else:
            corr = correct(output, target, topk=(1, 2))
            correct1 += corr[0]
            correct2 += corr[1]

        if batch_idx % log_interval == 0:
                tqdm.write(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}. '
                    'Top-1 accuracy: ({:.4f}%). '.format(epoch, batch_idx, len(loader),
                                                            100. * batch_idx / len(loader), loss.item(),
                                                            100. * correct1 / (batch_size * (batch_idx + 1))))
        
    return loss.item(), correct1 / len(loader.dataset), correct2

def test(model, loader, criterion, device, dtype):
    model.eval()
    test_loss = 0
    correct1, correct2 = 0, 0

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        with torch.no_grad():
            output = model(data)
            #test_loss += criterion(output, target).item()  # sum up batch loss
            corr = correct(output, target, topk=(1, 2))
        correct1 += corr[0]
        correct2 += corr[1]

    tqdm.write(
        '\nTest set:Top1: {}/{} ({:.4f}%), '
        'Top5: {}/{} ({:.2f}%)'.format(int(correct1), len(loader.dataset),
                                       100. * correct1 / len(loader.dataset), int(correct2),
                                       len(loader.dataset), 100. * correct2 / len(loader.dataset)))
    return test_loss, correct1 / len(loader.dataset), correct2 / len(loader.dataset)

def calculate_metrics(predicted_values,true_values):
    threshold_values = [i / 100 for i in range(101)]
    # -1, -2 are attacks, +1 is not an attack -> True when attack, False otherwise
    true_boolean_values = [int(x) == 1 for x in true_values]

    attack_count = true_boolean_values.count(False)
    bonafide_count = true_boolean_values.count(True)
    apcers = []
    bpcers = []
    acers = []

    for threshold in threshold_values:
        threshold_results = [x >= threshold for x in predicted_values]

        # APCER
        false_acceptance_rate = ((1 / bonafide_count) * sum(
            [1 for ex, res in zip(true_boolean_values, threshold_results) if ex and not res]))
        apcers.append(false_acceptance_rate)

        # BPCER
        false_rejection_rate = sum(
            [1 for ex, res in zip(true_boolean_values, threshold_results) if not ex and res]) / attack_count
        bpcers.append(false_rejection_rate)

        # ACER
        acer = (false_acceptance_rate + false_rejection_rate) / 2
        acers.append(acer)

    min_i = 0
    min_val = 100000
    for i, acer in enumerate(acers):
        if abs(apcers[i]-bpcers[i])< min_val:
            min_i = i
            min_val =abs(apcers[i]-bpcers[i])

    return {"apcer": apcers[min_i], "bpcer": bpcers[min_i], "acer": acers[min_i]},acers[min_i]

def val(model, loader, device, dtype):
    model.eval()
    preds_list = []
    label_list = []
    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data = data.to(device = device, dtype = dtype)
        target = target.to(device = device)

        with torch.no_grad():
            output = model(data) 
            output_soft = torch.softmax(output, dim = -1)
            
            preds = output_soft.to(device).detach().cpu().numpy()
            labels = target.to(device).detach().cpu().numpy()
            
            tmp_preds_list = []
            for n in range(len(preds)):
                preds_score = np.sum(preds[n][-fake_class_nums : ])
                labels[n] = (labels[n] >= fake_class_nums)
                tmp_preds_list.append(preds_score)
            preds_list.extend(tmp_preds_list)
            label_list.extend(labels)
    out,acer=calculate_metrics(preds_list,label_list)
    print(out)
    return acer

def train_network(start_epoch, epochs, scheduler, model, train_loader, test_loader, val_loader, optimizer, criterion, device, dtype,
                  batch_size, log_interval, csv_logger, save_path, claimed_acc1, claimed_acc5, best_test, best_loss, args):
    
    best_acer = val(model,val_loader, device, dtype)
    for epoch in trange(start_epoch, epochs + 1):
        train_loss, train_acc, train_acc_last, = train(model, train_loader, epoch, optimizer, criterion, device,
                                                              dtype, batch_size, log_interval, scheduler, args)
        test_loss, test_acc, test_acc_last = test(model, test_loader, criterion, device, dtype)
        print(str(test_acc) + " vs " + str(best_test))

        if test_acc >= best_test:
            best_test = test_acc
            acer = val(model, val_loader, device, dtype)

            save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_test,
                         'optimizer': optimizer.state_dict()}, acer <= best_acer, filepath=save_path)

            if acer <= best_acer:
                best_acer = acer

        csv_logger.write({'epoch': epoch + 1, 'val_error1': 1 - test_acc, 'val_error5': 1 - test_acc_last,
                          'val_loss': test_loss, 'train_error1': 1 - train_acc,
                          'train_error5': 1 - train_acc_last, 'train_loss': train_loss,'val_acer':best_acer})
        
        csv_logger.plot_progress(claimed_acc1=claimed_acc1, claimed_acc5=claimed_acc5,title=model_title)

        if not isinstance(scheduler, CyclicLR):
            scheduler.step()

    csv_logger.write_text('Best accuracy is {:.2f}% top-1'.format(best_test * 100.))

def main():
    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(1000, 100000)
    print("Get Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpus:
        torch.cuda.manual_seed_all(args.seed)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save == '':
        args.save = time_stamp
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.gpus is not None:
        if ',' not in args.gpus:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            args.gpus = [int(i) for i in args.gpus.split(',')]
            device = 'cuda:' + str(args.gpus[0])
            cudnn.benchmark = True
    else:
        device = 'cpu'

    if args.type == 'float64':
        dtype = torch.float64
    elif args.type == 'float32':
        dtype = torch.float32
    elif args.type == 'float16':
        dtype = torch.float16
    else:
        raise ValueError('Wrong type!')  # TODO int8


    model=timm.create_model(model_title, pretrained=True, num_classes=2)
    print(model)
    # for para in model.parameters():
    #     para.requires_grad = False
    # model.head[-1].weight.requires_grad = True
    num_parameters = sum([l.nelement() for l in model.parameters()])
    print('number of parameters: {}'.format(num_parameters))

    train_loader, test_loader, val_loader = get_loaders(args.dataroot, args.batch_size, args.batch_size, args.img_size,
                                           args.workers, data_val_path)

    # define loss function (criterion) and optimizer
    if args.mixup_prob > 0:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.label_smooth > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing = args.label_smooth)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.gpus is not None:
        model = torch.nn.DataParallel(model, args.gpus)
    model.to(device=device, dtype=dtype)
    criterion.to(device=device, dtype=dtype)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.learning_rate, momentum=args.momentum, weight_decay=args.decay,
                                nesterov=True)

    if args.find_clr:
        find_bounds_clr(model, train_loader, optimizer, criterion, device, dtype, min_lr=args.min_lr,
                        max_lr=args.max_lr, step_size=args.epochs_per_step * len(train_loader), mode=args.mode,
                        save_path=save_path)
        return

    if args.clr:
        print("Use CyclicLR!")
        scheduler = CyclicLR(optimizer, base_lr=args.min_lr, max_lr=args.max_lr,
                             step_size=args.epochs_per_step * len(train_loader), mode=args.mode)
    else:
        print("Use MultiStepLR!")
        scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)

    # optionally resume from a checkpoint
    data = None
    best_test = 0.8
    best_loss = 0.1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = args.start_epoch
            print(checkpoint.keys())
            #best_test = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        elif os.path.isdir(args.resume):
            checkpoint_path = os.path.join(args.resume, 'checkpoint.pth.tar')
            csv_path = os.path.join(args.resume, 'results.csv')
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location=device)
            args.start_epoch = args.start_epoch
            # best_test = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
            data = []
            with open(csv_path) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append(row)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        loss, top1, top5 = test(model, test_loader, criterion, device, dtype)  # TODO
        return

    csv_logger = CsvLogger(filepath=save_path, data=data)
    csv_logger.save_params(sys.argv, args)

    claimed_acc1 = None
    claimed_acc5 = None

    train_network(args.start_epoch, args.epochs, scheduler, model, train_loader, test_loader, val_loader, optimizer, criterion,
                  device, dtype, args.batch_size, args.log_interval, csv_logger, save_path, claimed_acc1, claimed_acc5,
                  best_test, best_loss, args)

if __name__ == '__main__':
    main()
