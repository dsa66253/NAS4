from __future__ import print_function
import math
import os
import sys
from pathlib import Path
from matplotlib.pyplot import plot
import torch
import torch.optim as optim
# import torch.backends.cudnn as cudnn
import argparse
from torch import nn
from torchvision import transforms, datasets
from data.config import cfg_newnasmodel
from tensorboardX import SummaryWriter
import numpy as np
# from models.newmodel_5cell import NewNasModel
from feature.learning_rate import adjust_learning_rate
from feature.normalize import normalize
from feature.resume_net import resumeNet
from feature.make_dir import makeDir
from feature.split_data import split_data
from feature.random_seed import set_seed_cpu
from PIL import ImageFile
from tqdm import tqdm
from models.mynewmodel_5cell import NewNasModel
from retrainModel import NewNasModel
from feature.utility import plot_acc_curve, setStdoutToFile, setStdoutToDefault
from feature.utility import getCurrentTime, accelerateByGpuAlgo, get_device, plot_loss_curve
# from train_nas_5cell import prepareDataloader
stdoutTofile = True
accelerateButUndetermine = cfg_newnasmodel["cuddbenchMark"]
recover = False
def parse_args(k):
    parser = argparse.ArgumentParser(description='imagenet nas Training')
    parser.add_argument('--network', default='newnasmodel', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='./retrainSavedModel/',
                        help='Location to save checkpoint models')
    parser.add_argument('--log_dir', default='./tensorboard_retrain_pdarts/',
                        help='Location to save logging')
    parser.add_argument('--decode_folder', type=str, default='./weights_pdarts_nodrop',
                        help='put decode folder')
    parser.add_argument('--genotype_file', type=str, default='genotype_' + str(k) + '.npy',
                        help='put decode file')
    parser.add_argument('--pltSavedDir', type=str, default='./plot',
                        help='plot train loss and val loss')
    parser.add_argument('--trainDataSetFolder', default='./dataset1/train',
                        help='training data set folder')
    parser.add_argument('--retrainSavedModel', default='./retrainSavedModel',
                    help='where to save retrain model')
    parser.add_argument('--savedCheckPoint', default='./savedCheckPoint',
                    help='where to save checkpoint model')
    args = parser.parse_args()
    return args

def prepareDataSet():
    #info prepare dataset
    print('Loading Dataset with seed_img {}...'.format(seed_img))
    train_transforms = normalize(seed_img, cfg['image_size'])  # 正規化照片
    try:
        all_data = datasets.ImageFolder(args.trainDataSetFolder, transform=train_transforms)
    except Exception as e:
        print("Fail to load data set from: ",  args.trainDataSetFolder)
        print(e)
        exit()
    train_data, val_data = split_data(all_data, 0.2)  # 切訓練集跟驗證集
    return train_data, val_data

def prepareDataLoader(trainData, valData):
    #info prepare dataloader
    train_loader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, num_workers=args.num_workers,
                                            shuffle=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valData, batch_size=batch_size, num_workers=args.num_workers,
                                            shuffle=False, pin_memory=True)
    return train_loader, val_loader

def prepareLossFunction():
    #info prepare loss function
    print('Preparing loss function...')
    return  nn.CrossEntropyLoss()

def prepareModel():
    #info check decode alphas and load it
    if os.path.isdir(args.decode_folder):
        try:
            genotype_filename = os.path.join(args.decode_folder, args.genotype_file)
            cell_arch = np.load(genotype_filename)
            print("successfully load decode alphas")
        except:
            print("Fail to load decode alpha")
    else:
        print('decode_folder does\'t exist!')
        sys.exit(0)
        
    #info prepare model
    print("Preparing model...")
    if cfg['name'] == 'NewNasModel':
        # set_seed_cpu(seed_weight) #* have already set seed in main function 
        net = NewNasModel(cfg["numOfLayers"], 
                        cfg["numOfInnerCell"], 
                        numOfClasses=cfg["numOfClasses"], 
                        cellArch=cell_arch)
        net.train()
        net = net.to(device)
        print("net.cellArchTrans:", net.cellArchTrans)
        print("net", net)
    return net
def prepareOpt(net):
    return optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum,
                    weight_decay=weight_decay)  # 是否采取 weight_decay
    
    
def saveCheckPoint(epoch, optimizer, net, lossRecord):
    makeDir(args.savedCheckPoint)
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': lossRecord,
            }, 
            os.path.join(args.savedCheckPoint, "{}_{}.pt".format(args.network, epoch)))
    except Exception as e:
        print("Failt to save check point")
        print(e)
def recoverFromCheckPoint(model, optimizer):
    pass
    checkpoint = torch.load(args.savedCheckPoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']

def printNetWeight(net):
    for name, para in net.named_parameters():
        print(name, para)
        

def myTrain(kth, trainData, trainDataLoader, valDataLoader, net, model_optimizer, criterion):
    print("start train kth={}...".format(kth))
    global last_epoch_val_acc #?幹嘛用
    ImageFile.LOAD_TRUNCATED_IMAGES = True#* avoid damage image file


    # print("Training with learning rate = %f, momentum = %f, lambda = %f " % (initial_lr, momentum, weight_decay))
    #info other setting
    writer = SummaryWriter(log_dir=args.log_dir,
                        comment="LR_%.3f_BATCH_%d".format(initial_lr, batch_size))
    
    epoch_size = math.ceil(len(trainData) / batch_size)#* It should be number of batch per epoch
    max_iter = cfg['epoch'] * epoch_size #* it's correct here. It's the totoal iterations.
    #* an iteration go through a mini-batch(aka batch)
    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0
    start_iter = 0
    epoch = 0
    record_train_loss = np.array([])
    record_val_loss = np.array([])
    record_train_acc = np.array([])
    record_val_acc = np.array([])
    print('Start to train...')
    
    #info start training loop
    for iteration in tqdm(range(start_iter, max_iter), unit =" iterations on {}".format(kth)):
        
        if (iteration % epoch_size == 0) and (iteration != 0):
            lossRecord = {"train": record_train_loss, "val": record_val_loss}
            saveCheckPoint(epoch, model_optimizer, net, lossRecord)
            epoch = epoch + 1
            if recover:
                net, model_optimizer, epoch, lossRecord = recoverFromCheckPoint(net, model_optimizer)
            
        #* finish an epoch
        if iteration % epoch_size == 0:
            print("start training epoch{}...".format(epoch))
            train_batch_iterator = iter(trainDataLoader)

        # lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)
        train_images, train_labels = next(train_batch_iterator)
        val_batch_iterator = iter(valDataLoader)
        val_images, val_labels = next(val_batch_iterator)
        # plot_img(train_images, train_labels, val_images, val_labels)
        train_images, train_labels = train_images.to(device), train_labels.to(device)
        val_images, val_labels = val_images.to(device), val_labels.to(device)

        
        # forward pass
        train_outputs = net(train_images)
        # calculate loss
        train_loss = criterion(train_outputs, train_labels)
        # backward pass
        model_optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        # update weight
        model_optimizer.step()
        if iteration % epoch_size == 0:
            with torch.no_grad():
                #info recording training accruacy
                _, predicts = torch.max(train_outputs.data, 1)
                record_train_loss = np.append(record_train_loss, train_loss.item())

                val_outputs = net(val_images)
                _, predicts_val = torch.max(val_outputs.data, 1)
                val_loss = criterion(val_outputs, val_labels)
                record_val_loss = np.append(record_val_loss, val_loss.item())


                # 計算訓練集準確度
                total_images = 0
                correct_images = 0
                total_images += train_labels.size(0)
                correct_images += (predicts == train_labels).sum()

                # 計算驗證集準確度
                total_images_val = 0
                correct_images_val = 0
                total_images_val += val_labels.size(0)
                correct_images_val += (predicts_val == val_labels).sum()

            # load_t1 = time.time()
            # batch_time = load_t1 - load_t0
            # eta = int(batch_time * (max_iter - iteration))
            # print(
            #     'Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || train_loss: {:.4f} || val_loss: {:.4f} || train_Accuracy: {:.4f} || val_Accuracy: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || '
            #     'ETA: {} '
            #         .format(epoch, max_epoch, (iteration % epoch_size) + 1,
            #                 epoch_size, iteration + 1, max_iter, train_loss.item(), val_loss.item(),
            #                 100 * correct_images.item() / total_images, 100 * correct_images_val.item() / total_images_val,
            #                 lr, batch_time, str(datetime.timedelta(seconds=eta))))
            trainAcc = correct_images / total_images
            valAcc = correct_images_val / total_images_val
            record_train_acc = np.append(record_train_acc, trainAcc.cpu())
            record_val_acc = np.append(record_val_acc, valAcc.cpu())
            writer.add_scalar('Train_Loss', train_loss.item(), iteration + 1)
            writer.add_scalar('Val_Loss', val_loss.item(), iteration + 1)
            writer.add_scalar('train_Acc', 100 * trainAcc, iteration + 1)
            writer.add_scalar('val_Acc', 100 * valAcc, iteration + 1)
            last_epoch_val_acc = 100 * correct_images_val / total_images_val

        
        
    lossRecord = {"train": record_train_loss, "val": record_val_loss}
    accRecord = {"train": record_train_acc, "val": record_val_acc}
    torch.save(net.state_dict(), os.path.join(args.save_folder, cfg['name'] + str(kth) + '_Final.pth'))
    return last_epoch_val_acc, lossRecord, accRecord


if __name__ == '__main__':
    device = get_device()
    print("running on device: {}".format(device))
    valList = []

    for k in range(3):
        #info handle stdout to a file
        if stdoutTofile:
            trainLogDir = "./log"
            makeDir(trainLogDir)
            f = setStdoutToFile(trainLogDir+"/retrain_5cell_{}th.txt".format(str(k)))
            
        accelerateByGpuAlgo(cfg_newnasmodel["cuddbenchMark"])
        #info set seed
        if k == 0:
            seed_img = 10
            seed_weight = 20
        elif k == 1:
            seed_img = 255
            seed_weight = 278
        else:
            seed_img = 830
            seed_weight = 953
        #! test same initial weight
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        args = parse_args(str(k))
        makeDir(args.save_folder)
        makeDir(args.log_dir)
        makeDir(args.retrainSavedModel)
        makeDir(args.pltSavedDir)
        

        cfg = None
        if args.network == "newnasmodel":
            cfg = cfg_newnasmodel
        else:
            print('Retrain Model %s doesn\'t exist!' % (args.network))
            sys.exit(0)
            
        batch_size = cfg['batch_size']

        #todo find what do these stuff do
        num_workers = args.num_workers
        momentum = args.momentum
        weight_decay = args.weight_decay
        initial_lr = args.lr
        gamma = args.gamma

        set_seed_cpu(seed_img)
        
        print("seed_img{}, seed_weight{} start at ".format(seed_img, seed_weight), getCurrentTime())
        print("cfg", cfg)
        trainData, valData = prepareDataSet()
        trainDataLoader, valDataLoader = prepareDataLoader(trainData, valData)
        criterion = prepareLossFunction()
        net = prepareModel()
        model_optimizer = prepareOpt(net)
        last_epoch_val_ac, lossRecord, accRecord = myTrain(k, trainData, trainDataLoader, valDataLoader, net, model_optimizer, criterion)  # 進入model訓練
        plot_loss_curve(lossRecord, "loss_{}".format(k), args.pltSavedDir)
        plot_acc_curve(accRecord, "acc_{}".format(k), args.pltSavedDir)
        valList.append(last_epoch_val_acc)
        print('retrain validate accuracy:', valList)
        #info handle output file
        if stdoutTofile:
            setStdoutToDefault(f)
    print('retrain validate accuracy:', valList)



