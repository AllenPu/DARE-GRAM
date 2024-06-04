import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import model
import transform as tran
import numpy as np
import os
import argparse
torch.set_num_threads(1)
import math
from read_data import ImageList
from tqdm import tqdm
import matplotlib.pyplot as  plt


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description='PyTorch DAregre experiment')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--src', type=str, default='rc', metavar='S',
                    help='source dataset')
parser.add_argument('--tgt', type=str, default='t', metavar='S',
                    help='target dataset')
parser.add_argument('--lr', type=float, default=0.1,
                        help='init learning rate for fine-tune')
parser.add_argument('--gamma', type=float, default=0.0001,
                        help='learning rate decay')
parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
parser.add_argument('--tradeoff', type=float, default=0.001,
                        help='tradeoff of RSD')
parser.add_argument('--tradeoff2', type=float, default=0.01,
                        help='tradeoff of BMP')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)


#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

data_transforms = {
    'train': tran.rr_train(resize_size=224),
    'val': tran.rr_train(resize_size=224),
    'test': tran.rr_eval(resize_size=224),
}
# set dataset
batch_size = {"train": 36, "val": 36, "test": 4}
rc="realistic.txt"
rl="real.txt"
t="toy.txt"

rc_t="realistic_test.txt"
rl_t="real_test.txt"
t_t="toy_test.txt"





def Regression_test(loader, model, src, tgt):
    MSE = [0, 0, 0]
    MAE = [0, 0, 0]
    number = 0
    upper , bottom, gt_upper, gt_bottom = [], [], [], []
    norm_list = []
    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(loader['test']):
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels1 = labels[:, 0]
            labels2 = labels[:, 1]
            labels1 = labels1.unsqueeze(1)
            labels2 = labels2.unsqueeze(1)
            labels = torch.cat((labels1, labels2), dim=1)
            labels = labels.float() / 39
            pred, feature = model(imgs)
            norms = torch.norm(feature, p = 'fro', dim=-1)
            norm_list.extend(norms.cpu().tolist())
            MSE[0] += torch.nn.MSELoss(reduction='sum')(pred[:, 0], labels[:, 0])
            MAE[0] += torch.nn.L1Loss(reduction='sum')(pred[:, 0], labels[:, 0])
            MSE[1] += torch.nn.MSELoss(reduction='sum')(pred[:, 1], labels[:, 1])
            MAE[1] += torch.nn.L1Loss(reduction='sum')(pred[:, 1], labels[:, 1])
            MSE[2] += torch.nn.MSELoss(reduction='sum')(pred, labels)
            MAE[2] += torch.nn.L1Loss(reduction='sum')(pred, labels)
            number += imgs.size(0)
            bottom.extend(pred[:, 0].cpu().tolist())
            upper.extend(pred[:, 1].cpu().tolist())
            gt_bottom.extend(labels[:, 0].cpu().tolist())
            gt_upper.extend(labels[:, 1].cpu().tolist())
    for j in range(3):
        MSE[j] = MSE[j] / number
        MAE[j] = MAE[j] / number
    print(f' source is {src} target is {tgt}')
    #
    m = sum(norm_list)/len(norm_list)
    print("\tMeanOfFrob : {0}\n".format(m))
    '''
    print("\tMSE : {0},{1}\n".format(MSE[0], MSE[1]))
    print("\tMAE : {0},{1}\n".format(MAE[0], MAE[1]))
    print("\tMSEall : {0}\n".format(MSE[2]))
    print("\tMAEall : {0}\n".format(MAE[2]))
    #
    #
    #x = [i for i in range(len(upper))]
    l1_upper = list(map(lambda x: x[0]-x[1], zip(upper, gt_upper)))
    l1_bottom = list(map(lambda x: x[0]-x[1], zip(bottom, gt_bottom)))
    #plt.plot(x, gt_upper, label='labels')
    #plt.plot(x, upper, label='upper_preds')
    plt.hist(l1_upper, label='upper_differrence')
    plt.legend()
    #plt.draw()
    plt.savefig('./imgs/s_s/pic-hist-{}_src-{}-tgt-{}.png'.format('upper', src, tgt))
    plt.close()
    plt.hist(l1_bottom, label='bottom_difference')
    #plt.plot(x, gt_bottom, label='labels')
    #plt.plot(x, bottom, label='bottom_preds')
    plt.legend()
    #lt.draw()
    plt.savefig('./imgs/s_s/pic-hist-{}_src-{}-tgt-{}.png'.format('bottom', src, tgt))
    plt.close()
    '''


class Model_Regression(nn.Module):
    def __init__(self):
        super(Model_Regression,self).__init__()
        self.model_fc = model.Resnet18Fc()
        self.classifier_layer = nn.Linear(512, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        self.classifier_layer = nn.Sequential(self.classifier_layer,  nn.Sigmoid())
        self.predict_layer = nn.Sequential(self.model_fc,self.classifier_layer)
    def forward(self,x):
        feature = self.model_fc(x)
        outC= self.classifier_layer(feature)
        return(outC,feature)




models = ['source_c_target_n.pth', 'source_c_target_s.pth', 'source_n_target_c.pth', 'source_n_target_s.pth', 'source_s_target_c.pth', 'source_s_target_n.pth']
sources = ['c', 'c', 'n', 'n', 's', 's']
targets = ['n', 's', 'c', 's', 'c', 'n']

c_t="color_test.txt"
n_t="noisy_test.txt"
s_t="scream_test.txt"

path_test = '/home/rpu2/scratch/code/dsprites-dataset/da'



for i, (m, sour, tar) in enumerate(zip(models, sources, sources)):
    if tar == 'n':
        target_path_t = "color_test.txt"
    if tar == 'c':
        target_path_t = "noisy_test.txt"
    if tar == 's':
        target_path_t = "scream_test.txt"
    dsets = {"test": ImageList(open(target_path_t).readlines(), img_path=path_test, transform=data_transforms["test"])}
    dset_loaders = {}
    dset_loaders["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size=batch_size["test"],
                                                   shuffle=False, num_workers=16)

    dset_sizes = {x: len(dsets[x]) for x in ['test']}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Model_R = torch.load(m)
    #
    Model_R.eval()
    #Regression_test(dset_loaders, Model_R.predict_layer, src=sour, tgt=tar)
    Regression_test(dset_loaders, Model_R, src=sour, tgt=tar)



for i, (m, sour, tar) in enumerate(zip(models, sources, targets)):
    if tar == 'n':
        target_path_t = "color_test.txt"
    if tar == 'c':
        target_path_t = "noisy_test.txt"
    if tar == 's':
        target_path_t = "scream_test.txt"
    dsets = {"test": ImageList(open(target_path_t).readlines(), img_path=path_test, transform=data_transforms["test"])}
    dset_loaders = {}
    dset_loaders["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size=batch_size["test"],
                                                   shuffle=False, num_workers=16)

    dset_sizes = {x: len(dsets[x]) for x in ['test']}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Model_R = torch.load(m)
    #
    Model_R.eval()
    #Regression_test(dset_loaders, Model_R.predict_layer, src=sour, tgt=tar)
    Regression_test(dset_loaders, Model_R, src=sour, tgt=tar)




