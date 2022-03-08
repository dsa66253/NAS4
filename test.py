# 测试集
import argparse
import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import Subset
from torchvision import transforms, datasets
from data.config import cfg_newnasmodel
# from models.newmodel_8cell import NewNasModel
from models.newmodel_5cell import NewNasModel
from feature.learning_rate import adjust_learning_rate
from feature.normalize import normalize
from feature.make_dir import makeDir
from feature.random_seed import set_seed_cpu, set_seed_gpu


def parse_args(i):
    parser = argparse.ArgumentParser(description='imagenet nas Training')
    parser.add_argument('-m', '--trained_model',
                        default='./weights_retrain_pdarts/NewNasModel' + str(i) + '_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='newnasmodel', help='Backbone network mobile0.25 or resnet50')
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
    parser.add_argument('--genotype_file', type=str, default='genotype_' + str(i) + '.npy',
                        help='put decode file')
    args = parser.parse_args()
    return args


def test(seed_cpu):
    PATH_test = r"./dataset1/test"
    test = Path(PATH_test)
    test_transforms = normalize(seed_cpu, img_dim)

    # choose the training datasets
    test_data = datasets.ImageFolder(test, transform=test_transforms)

    print("test_data.class_to_idx", test_data.class_to_idx)

    # prepare data loaders (combine dataset and sampler)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    print("test_data length: ", len(test_data))

    if os.path.isdir(args.decode_folder):
        genotype_filename = os.path.join(args.decode_folder, args.genotype_file)
        cell_arch = np.load(genotype_filename)
        print(cell_arch)
        print('Load best alpha for each cells from %s' % (genotype_filename))
    else:
        print('Decode path is not exist!')
        sys.exit(0)

    def check_keys(model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def remove_prefix(state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def load_model(model, pretrained_path, load_to_cpu):
        print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = remove_prefix(pretrained_dict, 'module.')
        check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    # net and model
    print("NewNasModel")
    net = NewNasModel(num_classes=num_classes, cell_arch=cell_arch, num_cells=num_cell)
    print("load_model")
    net = load_model(net, args.trained_model, args.cpu)
    print("eval")
    net.eval()
    print('Finished loading model!')
    # print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    confusion_matrix_torch = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(test_loader, 0):
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
    for i in range(3):
        args = parse_args(str(i))
        # makeDir(args.save_folder, args.log_dir)
        cfg = None
        if args.network == "newnasmodel":
            cfg = cfg_newnasmodel
        else:
            print('Retrain Model %s doesn\'t exist!' % (args.network))
            sys.exit(0)

        num_classes = 10
        num_cell = 5

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
        test(seed_cpu)
