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


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description='PyTorch DAregre experiment')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--src', type=str, default='c', metavar='S',
                    help='source dataset')
parser.add_argument('--tgt', type=str, default='n', metavar='S',
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
parser.add_argument('--batch', type=int, default=36,
                        help='batch size')
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
batch_size = {"train": args.batch, "val": args.batch, "test": 36}
c="color.txt"
n="noisy.txt"
s="scream.txt"

c_t="color_test.txt"
n_t="noisy_test.txt"
s_t="scream_test.txt"

if args.src =='c':
    source_path = c
elif args.src =='n':
    source_path = n 
elif args.src =='s':
    source_path = s
img_path_source = '/home/rpu2/scratch/code/dsprites-dataset/da'

if args.tgt =='c':
    target_path = c_t
elif args.tgt =='n':
    target_path = n_t
elif args.tgt =='s':
    target_path = s_t
img_path_target = '/home/rpu2/scratch/code/dsprites-dataset/da'



def Regression_test(loader, model):
    MSE = [0, 0, 0]
    MAE = [0, 0, 0]
    number = 0
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
            pred = model(imgs)
            MSE[0] += torch.nn.MSELoss(reduction='sum')(pred[:, 0], labels[:, 0])
            MAE[0] += torch.nn.L1Loss(reduction='sum')(pred[:, 0], labels[:, 0])
            MSE[1] += torch.nn.MSELoss(reduction='sum')(pred[:, 1], labels[:, 1])
            MAE[1] += torch.nn.L1Loss(reduction='sum')(pred[:, 1], labels[:, 1])
            MSE[2] += torch.nn.MSELoss(reduction='sum')(pred, labels)
            MAE[2] += torch.nn.L1Loss(reduction='sum')(pred, labels)
            number += imgs.size(0)
    for j in range(3):
        MSE[j] = MSE[j] / number
        MAE[j] = MAE[j] / number
    print("\tMSE : {0},{1}\n".format(MSE[0], MSE[1]))
    print("\tMAE : {0},{1}\n".format(MAE[0], MAE[1]))
    print("\tMSEall : {0}\n".format(MSE[2]))
    print("\tMAEall : {0}\n".format(MAE[2]))


store_name = f'source_{args.src}_target_{args.tgt}'

dsets = {"train": ImageList(open(source_path).readlines(), img_path=img_path_source, transform=data_transforms["train"]), \
         "test": ImageList(open(target_path).readlines(), img_path=img_path_target, transform=data_transforms["test"])}

dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size[x],
                                               shuffle=True, num_workers=8)
                for x in ['train']}
dset_loaders["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size=batch_size["test"],
                                                   shuffle=False, num_workers=16)

dset_sizes = {x: len(dsets[x]) for x in ['train', 'test']}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001, weight_decay=0.0005):
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        param_group['weight_decay'] = weight_decay * 2
        i += 1
    return optimizer

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



Model_R = Model_Regression()
Model_R = Model_R.to(device)

Model_R.train(True)
criterion = {"regressor": nn.MSELoss()}
optimizer_dict = [{"params": filter(lambda p: p.requires_grad, Model_R.model_fc.parameters()), "lr": 0.1},
                  {"params": filter(lambda p: p.requires_grad, Model_R.classifier_layer.parameters()), "lr": 1}]
optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
train_cross_loss = train_rsd_loss = train_total_loss = 0.0
len_source = len(dset_loaders["train"]) - 1
param_lr = []
iter_source = iter(dset_loaders["train"])


for param_group in optimizer.param_groups:
    param_lr.append(param_group["lr"])
test_interval = 500
num_iter = 10002
for iter_num in tqdm(range(1, num_iter + 1)):
    Model_R.train(True)
    optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr, gamma=args.gamma, power=0.75,
                                 weight_decay=0.0005)
    optimizer.zero_grad()
    if iter_num % len_source == 0:
        iter_source = iter(dset_loaders["train"])
    #if iter_num % len_target == 0:
    #    iter_target = iter(dset_loaders["val"])
    data_source = next(iter_source)#.__next__()
    #data_target = iter_target.next()
    inputs_source, labels_source = data_source
    labels1 = labels_source[:,0]
    labels2 = labels_source[:,1]
    labels1 = labels1.unsqueeze(1)
    labels2 = labels2.unsqueeze(1)
    labels_source = torch.cat((labels1,labels2),dim=1)
    labels_source = labels_source.float()/39
    #inputs_target, labels_target = data_target
    #inputs = torch.cat((inputs_source, inputs_target), dim=0)
    #inputs = inputs.to(device)
    labels = labels_source.to(device)
    inputs_s = inputs_source.narrow(0, 0, batch_size["train"]).to(device)
    #inputs_t = inputs.narrow(0, batch_size["train"], batch_size["train"])
    outC_s, feature_s = Model_R(inputs_s)
    #outC_t, feature_t = Model_R(inputs_t)
    classifier_loss = criterion["regressor"](outC_s, labels)
    #rsd_loss = RSD(feature_s,feature_t)
    total_loss = classifier_loss #+ args.tradeoff*rsd_loss
    total_loss.backward()
    optimizer.step()
    train_cross_loss += classifier_loss.item()
    #train_rsd_loss += rsd_loss.item()
    train_total_loss += total_loss.item()
    if iter_num % 500 == 0:
        print("Iter {:05d}, Average Cross Entropy Loss: {:.4f}; Average RSD Loss: {:.4f};  Average Training Loss: {:.4f}".format(
            iter_num, train_cross_loss / float(test_interval), train_rsd_loss / float(test_interval),
            train_total_loss / float(test_interval)))
        train_cross_loss = train_rsd_loss = train_total_loss  = 0.0
    if (iter_num % test_interval) == 0:
        Model_R.eval()
        Regression_test(dset_loaders, Model_R.predict_layer)
torch.save(Model_R, f'{store_name}.pth')

