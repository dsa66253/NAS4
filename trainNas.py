import os
import sys
import torch
import argparse
import torch.optim as optim


import torch.nn as nn
import math
import time
from torchvision import datasets
from data.config import cfg_nasmodel, cfg_alexnet
from models.alexnet import Baseline
# from tensorboardX import SummaryWriter #* how about use tensorbaord instead of tensorboardX
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from feature.normalize import normalize
from feature.make_dir import makeDir
from feature.split_data import split_data
from feature.random_seed import set_seed_cpu
from PIL import ImageFile
from tqdm import tqdm
from model import Model
from data.config import epoch_to_drop
from feature.utility import getCurrentTime, setStdoutToDefault, setStdoutToFile, accelerateByGpuAlgo
from feature.utility import plot_acc_curve, plot_loss_curve
stdoutTofile = True
accelerateButUndetermine = False
recover = False
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(description='imagenet nas Training')
    parser.add_argument('--network', default='nasmodel', help='Backbone network alexnet or nasmodel')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--nas_lr', '--nas-learning-rate', default=3e-3, type=float,
                        help='initial learning rate for nas optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='./weights_pdarts_nodrop/',
                        help='Location to save checkpoint models')
    parser.add_argument('--log_dir', default='./tensorboard_pdarts_nodrop/',
                        help='Location to save logging')
    parser.add_argument('--trainDataSetFolder', default='./dataset1/train',
                        help='training data set folder')
    parser.add_argument('--nasSavedModel', default='./nasSavedModel',
                        help='Where to save nas Model')
    parser.add_argument('--retrainSavedModel', default='./retrainSavedModel',
                    help='where to save retrain model')
    parser.add_argument('--savedCheckPoint', default='./savedCheckPoint',
                    help='where to save checkpoint model')
    parser.add_argument('--pltSavedDir', type=str, default='./plot',
                        help='plot train loss and val loss')
    parser.add_argument('--accLossDir', type=str, default='./accLoss',
                        help='plot train loss and val loss')
    args = parser.parse_args()
    return args

def prepareDataSet():
    #info prepare dataset
    print('Loading Dataset with seed_img {}...'.format(seed_img))
    train_transforms = normalize(seed_img, img_dim)  # 正規化照片
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
    print('Preparing loss function...')
    return  nn.CrossEntropyLoss()

def prepareModel():
    #info prepare model
    print("Preparing model...")
    if cfg['name'] == 'alexnet':
        # alexnet model
        net = Baseline(cfg["numOfClasses"])
        net = net.to(device)
        net.train()
    elif cfg['name'] == 'NasModel':
        # nas model
        # todo why pass no parameter list to model, and we got cfg directly in model.py from config.py
        net = Model()
        print("net", net)
        #! move to cuda before assign net's parameters to optim, otherwise, net on cpu will slow down training speed
        net = net.to(device)
        net.train()
    return net

def prepareOpt(net):
    #info prepare optimizer
    print("Preparing optimizer...")
    if cfg['name'] == 'alexnet':  # BASELINE
        optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
        return optimizer
    elif cfg['name'] == 'NasModel':
        model_optimizer = optim.SGD(net.getWeight(), lr=initial_lr, momentum=momentum,
                                    weight_decay=weight_decay)
        nas_optimizer = optim.Adam(net.getAlphas(), lr=nas_initial_lr, weight_decay=weight_decay)
        return model_optimizer, nas_optimizer
    
def printNetWeight(net):
    for name, para in net.named_parameters():
        print(name, para)


    
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

def saveAccLoss(lossRecord, accRecord):
    try:
        np.save(os.path.join(args.accLossDir, "trainLoss"), lossRecord["train"])
        np.save(os.path.join(args.accLossDir, "valnLoss"), lossRecord["val"])
        np.save(os.path.join(args.accLossDir, "trainAcc"), accRecord["train"])
        np.save(os.path.join(args.accLossDir, "trainVal"), accRecord["val"])
    except Exception as e:
        print("Fail to save acc and loss")
        print(e)

def myTrain(kth, trainData, train_loader, val_loader, net, model_optimizer, nas_optimizer, criterion):
    
    # calculate how many iterations
    epoch_size = math.ceil(len(trainData) / batch_size)#* It should be number of batch per epoch
    max_iter = max_epoch * epoch_size #* it's correct here. It's the totoal iterations.
    #* an iteration go through a mini-batch(aka batch)
    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0
    start_iter = 0
    epoch = 0
    
    # other setting
    # writer = SummaryWriter()
    writer = SummaryWriter(log_dir=args.log_dir,
            comment="LR_%.3f_BATCH_%d".format(initial_lr, batch_size))
    print("start to train...")
    
    record_train_loss = np.array([])
    record_val_loss = np.array([])
    record_train_acc = np.array([])
    record_val_acc = np.array([])
    
    
    
    print("minibatch size: ", epoch_size)
    #info start training loop
    for iteration in tqdm(range(start_iter, max_iter), unit =" iterations on {}".format(kth)):
        
        if (iteration % epoch_size == 0) and (iteration != 0):
            lossRecord = {"train": record_train_loss, "val": record_val_loss}
            saveCheckPoint(epoch, model_optimizer, net, lossRecord)
            epoch = epoch + 1
            if recover:
                net, model_optimizer, epoch, lossRecord = recoverFromCheckPoint(net, model_optimizer)
        
        # finish an epoch
        # print("begin iteration", iteration, " net.Alphas", net.alphas)
        if iteration % epoch_size == 0:
            print("start training epoch{}...".format(epoch))
            # create batch iterator
            # print("\ncurrent alphas id {} at epoch{}".format(id(net.alphas), epoch))
            train_batch_iterator = iter(train_loader)
            net.saveAlphas(epoch, kth)
            net.saveMask(epoch, kth)
            if epoch >= cfg['start_train_nas_epoch']:
                net.normalizeAlphas()
            if epoch in epoch_to_drop:
                net.dropMinAlpha()
            # 每10個EPOCH存一次權重
            if (epoch % 10 == 0 and epoch > 0):
                torch.save(
                    net.state_dict(),
                    os.path.join(
                        args.nasSavedModel,
                        cfg['name'] + '_' + str(kth) + '_epoch_' + str(epoch) + '.pth'
                    ),
                )

        load_t0 = time.time()
        

        # load train data
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        train_images, train_labels = next(train_batch_iterator)
        val_batch_iterator = iter(val_loader)
        val_images, val_labels = next(val_batch_iterator)

        train_images = train_images.to(device)
        train_labels = train_labels.to(device)
        val_images = val_images.to(device)
        val_labels = val_labels.to(device)

        model_optimizer.zero_grad()
        nas_optimizer.zero_grad()
        
        # Forward pass
        train_outputs = net(train_images, epoch, kth)
        # calculate loss
        train_loss = criterion(train_outputs, train_labels)
        # backward pass
        train_loss.backward()
        # print("epoch >= cfg['start_train_nas_epoch']", epoch >= cfg['start_train_nas_epoch'])
        # print("(epoch - cfg['start_train_nas_epoch']) % 2 == 0", (epoch - cfg['start_train_nas_epoch']) % 2 == 0)
        # print("epoch", epoch, "cfg['start_train_nas_epoch']", cfg['start_train_nas_epoch'])
        
        # take turns to optimize weight and alphas
        if epoch >= cfg['start_train_nas_epoch']:
            if (epoch - cfg['start_train_nas_epoch']) % 2 == 0:
                nas_optimizer.step()
            else:
                model_optimizer.step()
        else:
            model_optimizer.step()

            
        #info recording training process
        # model預測出來的結果 (訓練集)
        _, predicts = torch.max(train_outputs.data, 1)
        # record_train_loss.append(train_loss.item())
        
        #! Why she use validation directly at the end of an iteration.
        #! Usually we use validation after finishing all training.
        #! And chose the model generate with best accuracy on validation set
        # model預測出來的結果 (測試集)
        if (iteration % epoch_size == 0):
            val_outputs = net(val_images, epoch, kth)
            _, predicts_val = torch.max(val_outputs.data, 1)
            record_train_loss = np.append(record_train_loss, train_loss.item())
            val_loss = criterion(val_outputs, val_labels)
            record_val_loss = np.append(record_val_loss, val_loss.item())
            # print("end iteration", iteration, " net.Alphas", net.alphas)
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

            load_t1 = time.time()
            batch_time = load_t1 - load_t0
            eta = int(batch_time * (max_iter - iteration))
            # print(
            #     'Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || train_loss: {:.4f} || val_loss: {:.4f} || train_Accuracy: {:.4f} || val_Accuracy: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || '
            #     'ETA: {} '
            #         .format(epoch, max_epoch, (iteration % epoch_size) + 1,
            #                 epoch_size, iteration + 1, max_iter, train_loss.item(), val_loss.item(),
            #                 100 * correct_images.item() / total_images,
            #                 100 * correct_images_val.item() / total_images_val,
            #                 0.02, batch_time, str(datetime.timedelta(seconds=eta))))

            # 使用tensorboard紀錄LOSS、ACC
            # writer.add_scalar("test/1", iteration*2, iteration)
            
            
            trainAcc = correct_images / total_images
            valAcc = correct_images_val / total_images_val
            record_train_acc = np.append(record_train_acc, trainAcc.cpu())
            record_val_acc = np.append(record_val_acc, valAcc.cpu())
            
            writer.add_scalar('Train_Loss/k='+str(kth), train_loss.item(), iteration + 1)
            writer.add_scalar('Val_Loss/k='+str(kth), val_loss.item(), iteration + 1)
            writer.add_scalar('train_Acc/k='+str(kth), 100 * trainAcc, iteration + 1)
            writer.add_scalar('val_Acc/k='+str(kth), 100 * valAcc, iteration + 1)
            last_epoch_val_acc = 100 * correct_images_val / total_images_val
        if iteration==70:
            pass
            # exit()
    lossRecord = {"train": record_train_loss, "val": record_val_loss}
    accRecord = {"train": record_train_acc, "val": record_val_acc}
    saveAccLoss(lossRecord, accRecord)
    torch.save(net.state_dict(), os.path.join(args.nasSavedModel, cfg['name'] + str(kth) + '_Final.pth'))
    writer.close()
    
    return last_epoch_val_acc, lossRecord, accRecord





if __name__ == '__main__':
    device = get_device()
    torch.device(device)
    print("running on device: {}".format(device))
    torch.set_printoptions(precision=6, sci_mode=False, threshold=1000)
    torch.set_default_dtype(torch.float32) #* torch.float will slow the training speed
    valList = []
    exit()
    for k in range(3):
        #info set stdout to file
        if stdoutTofile:
            trainLogDir = "./log"
            makeDir(trainLogDir)
            f = setStdoutToFile(trainLogDir+"/train_nas_5cell_py_{}th.txt".format(str(k)))
        #info diifferent seeds fro different initail weights
        if k == 0:
            seed_img = 10
            seed_weight = 20
        elif k == 1:
            seed_img = 255
            seed_weight = 278
        else:
            seed_img = 830
            seed_weight = 953
        
        accelerateByGpuAlgo(accelerateButUndetermine)
        set_seed_cpu(seed_weight)  # 控制照片批次順序
        
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'
        args = parse_args()
        makeDir(args.save_folder) # refer to feature package
        makeDir(args.log_dir)
        makeDir(args.nasSavedModel)
        makeDir(args.savedCheckPoint)
        makeDir(args.accLossDir)
        cfg = None
        if args.network == "alexnet":
            cfg = cfg_alexnet
        elif args.network == "nasmodel":
            cfg = cfg_nasmodel
        else:
            print('Model %s doesn\'t exist!' % (args.network))
            sys.exit(0)
        print("cfg", cfg)
        img_dim = cfg['image_size']
        num_gpu = cfg['ngpu']
        batch_size = cfg['batch_size']
        max_epoch = cfg['epoch']
        gpu_train = cfg['gpu_train']

        num_workers = args.num_workers
        momentum = args.momentum
        weight_decay = args.weight_decay
        initial_lr = args.lr
        nas_initial_lr = args.nas_lr
        gamma = args.gamma
        save_folder = args.save_folder
        

        print("seed_img {}, seed_weight {} start at ".format(seed_img, seed_weight), getCurrentTime())
        print("training cfg", cfg)
        trainData, valData = prepareDataSet()
        trainDataLoader, valDataLoader = prepareDataLoader(trainData, valData)
        criterion = prepareLossFunction()
        net = prepareModel()
        model_optimizer, nas_optimizer = prepareOpt(net)
        last_epoch_val_ac, lossRecord, accRecord  = myTrain(k, trainData, trainDataLoader, valDataLoader, net, model_optimizer, nas_optimizer, criterion)  # 進入model訓練
        plot_loss_curve(lossRecord, "loss_{}".format(k), args.pltSavedDir)
        plot_acc_curve(accRecord, "acc_{}".format(k), args.pltSavedDir)
        valList.append(last_epoch_val_ac)
        print('train validate accuracy:', valList)
        if stdoutTofile:
            setStdoutToDefault(f)
        # exit() #* for examine why same initial value will get different trained model
    print('train validate accuracy:', valList)
        
        
        
        
        
        
        
        
        
        
        