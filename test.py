# 测试集
import argparse
from tqdm import tqdm
import os
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import Subset
from torchvision import transforms, datasets
from data.config import cfg_newnasmodel, cfg_alexnet
# from models.newmodel_8cell import NewNasModel
# from models.newmodel_5cell import NewNasModel
# from models.mynewmodel_5cell import NewNasModel
from retrainModel import NewNasModel
from models.alexnet import Baseline
from feature.learning_rate import adjust_learning_rate
from feature.normalize import normalize
from feature.make_dir import makeDir
from feature.utility import getCurrentTime, getCurrentTime1, setStdoutToDefault, setStdoutToFile, accelerateByGpuAlgo
from feature.random_seed import set_seed_cpu, set_seed_gpu


stdoutTofile = True
accelerateButUndetermine = True
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args(i):
    parser = argparse.ArgumentParser(description='imagenet nas Training')
    parser.add_argument('-m', '--trained_model',
                        default='./weights_retrain_pdarts/NewNasModel' + str(i) + '_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='newnasmodel', help='alexnet or newnasmodel')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--decode_folder', type=str, default='./weights_pdarts_nodrop',
                        help='put the path to resuming file if needed')
    parser.add_argument('--retrainSavedModel', type=str, default='./retrainSavedModel',
                        help='put the path to resuming file if needed')
    parser.add_argument('--genotype_file', type=str, default='genotype_' + str(i) + '.npy',
                        help='put decode file')
    args = parser.parse_args()
    return args

def printNetWeight(net):
    for name, para in net.named_parameters():
        print(name, para)
def prepareData():
    print("preparing data set")
    PATH_test = r"./dataset1/test"
    test = Path(PATH_test)
    test_transforms = normalize(seed_cpu, img_dim)

    # choose the training datasets
    test_data = datasets.ImageFolder(test, transform=test_transforms)
    print("test_data.class_to_idx", test_data.class_to_idx)
    return test_data

def prepareDataLoader(test_data):
    print("preparing data loader")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return test_loader
def prepareModel(num_classes, kth):
    print("preparing model: ", args.network)
    #info preparing alexnet model
    if args.network == "alexnet":
        try:
            net = Baseline(num_classes)
            modelLoadPath = os.path.join(args.retrainSavedModel, "alexnet{}_Final.pth".format(kth))
            net.load_state_dict(torch.load( modelLoadPath ))
            net = net.to(device)
            net.eval()
            print("Loading model from ", modelLoadPath)
            
            return net
        except:
            print("Fail to load model from ", modelLoadPath)
            exit()
    #info preparing trained NAS model

    if args.network == "newnasmodel":
        try :
            #info prepare architecture
            # os.path.isdir(args.decode_folder)
            genotype_filename = os.path.join(args.decode_folder, args.genotype_file)
            cell_arch = np.load(genotype_filename)
            print('Load best alpha for each cells from %s' % (genotype_filename))
            print(cell_arch)
        except:
            print("Fail to load architecture from ", genotype_filename)
            exit()
            
        try:
            #info prepare model
            net = NewNasModel(5, 1, numOfClasses=num_classes, cellArch=cell_arch)
            print("net ", net)
            modelLoadPath = os.path.join( args.retrainSavedModel, "NewNasModel{}_Final.pth".format(kth) )
            net.load_state_dict( torch.load( modelLoadPath ) )
            net = net.to(device)
            net.eval()
            print("Loading model from ", modelLoadPath)
            return net
        # todo test.py go wrong, check pytorch document how to test model
        except Exception as e:
            print("Fail to load model from ", modelLoadPath)
            print(e)
            exit()

def test(seed_cpu, test_loader, net):
    print("start testing")
    confusion_matrix_torch = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(tqdm(test_loader, 0)):
            images, labels = data
            labels = Variable(labels.cuda())
            images = Variable(images.cuda())
            outputs = net(images)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum()
            for t, p in zip(labels.view(-1), predict.view(-1)):
                confusion_matrix_torch[t.long(), p.long()] += 1
    # print(confusion_matrix_torch)
    print('Accuracy of the network on the 1500 test images: %f %%' % (100 * correct / total))


if __name__ == '__main__':
    
    device = get_device()
    for kth in range(3):
        #info handle stdout to a file
        if stdoutTofile:
            trainLogDir = "./log"
            makeDir(trainLogDir)
            f = setStdoutToFile(trainLogDir+"/test_py_{}th.txt".format(str(kth)))
        accelerateByGpuAlgo(accelerateButUndetermine)
        args = parse_args(str(kth))
        # makeDir(args.save_folder, args.log_dir)
        cfg = None
        try:
            if args.network == "newnasmodel":
                cfg = cfg_newnasmodel
            elif args.network == "alexnet":
                cfg = cfg_alexnet
        except:
            print('Retrain Model %s doesn\'t exist!' % (args.network))
            sys.exit(0)

        num_classes = 10

        img_dim = cfg['image_size']
        num_gpu = cfg['ngpu']
        batch_size = cfg['batch_size']
        max_epoch = cfg['epoch']
        gpu_train = cfg['gpu_train']

        num_workers = args.num_workers
        momentum = args.momentum
        weight_decay = args.weight_decay
        initial_lr = args.lr
        gamma = args.gamma

        seed_cpu = 28
        set_seed_cpu(seed_cpu)
        testSet = prepareData()
        testDataLoader = prepareDataLoader(testSet)
        net = prepareModel(num_classes, kth)
        printNetWeight(net)
        test(seed_cpu, testDataLoader, net)
        
        if stdoutTofile:
            setStdoutToDefault(f)
