#!/usr/bin/env python
# coding: utf-8

# # log io

# In[1]:

dvc = input("which device")
from datetime import datetime
import logging,os
from munch import Munch
args = {
    "save_root" : "./ckpt",
    "lr" : 0.00001,
    "epochs" : 100,
    "batch_size" : 32,
    "eps" : 0.02,
    "M_EPS" : 1e-16,
    "max_iter" : 1000,
    "err_bound" : 1e-8,
    "uot" : False,  
    "warm_up" : True,
    "save_dir" : None,
    "opt_interval" : 1,
    "opt_times" : 10,
    "end_times" : 100000,
    "resume" : "/home/home/weiyupeng/thome/ym/pred_saved/pred_net1123-164037.pth",
    "train_notes": "train_mlp with zero mean()"
}
args = Munch(args)

def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
time_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')

if args.get('save_dir',None) is None:# not set save dir
    args.save_dir = os.path.join(args.save_root, time_str)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
logger = get_logger(os.path.join(args.save_dir, 'train-{:s}.log'.format(time_str)))
os.system("cp {} {}".format(os.path.realpath(__file__), args.save_dir))
logger.info("cp {} {}".format(os.path.realpath(__file__), args.save_dir))
# In[2]:



# learning parameters
args.epochs =epochs = 100
batch_size = args.batch_size 
eps = args.eps
M_EPS = args.M_EPS
max_iter = args.max_iter
ep = args.err_bound
uot = args.uot
warm_up = args.warm_up

for k, v in args.items():
        logger.info("{}:\t{}".format(k.ljust(15), v))


# # dataset loader

# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib, scipy
from torch.utils.data import DataLoader
matplotlib.style.use('ggplot')
# initialize the computation device
device = torch.device('cuda:{}'.format(str(dvc)) if torch.cuda.is_available() else 'cpu')
C = torch.asarray(scipy.io.loadmat("/home/home/weiyupeng/thome/ym/ground-metric.mat")["M"]).to(device)
K = torch.exp(-C/eps)
top_k_matrix = C 


# In[4]:


# k cost
def top_k_cost(k,a,gt,dis_matrix):
    ind = torch.topk(a, k).indices
    b, cnum = a.shape
    sum = 0
    for i in range(b): # iter among batch
        mask = torch.nonzero(gt[i])
        dis_matrix_ = dis_matrix[mask,:].squeeze(1)
        sum += torch.sum(torch.min(dis_matrix_[:,ind[i]],dim = 0).values)
    return (1/(k*b) * sum).item()
# a = torch.randn(32,1000)
# gt = torch.ones(32,1000)
# print(top_k_cost(2,a,top_k_matrix))
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def multi_label_auc(y_true, y_pred):
    """
    计算多标签AUC的函数
    :param y_true: 真实标签，形状为[N, num_classes]
    :param y_pred: 预测标签，形状为[N, num_classes]
    :return: 多标签AUC
    """
    # 将标签转换为numpy数组
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    # 初始化多标签AUC值
    total_auc = 0.

    # 计算每个标签的AUC值，并对所有标签的AUC值求平均
    for i in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            auc = 0.5  # 如果标签中只有一个类别，则返回0.5
        total_auc += auc

    multi_auc = total_auc / y_true.shape[1]

    return multi_auc


# # 假设有3个标签和10个样本
# num_classes = 1000
# N = 10

# # 随机生成真实标签和预测标签
# y_true = torch.randint(0, 2, size=(N, num_classes)).float()
# y_pred = torch.randn(N, num_classes)

# # 计算多标签AUC
# multi_auc = multi_label_auc(y_true, y_pred)
# print('多标签AUC:', multi_auc)


# # OT and UOT solver

# In[5]:


# loss 的选项：
# dual_obj
# norm(grad) acutally norm(b-b0)


# In[27]:


# OT solver
import time
class ot_solver():
    def __init__(self, reg, K,device):
        self.reg = self.epsilon = reg
        self.K = K
        self.device = device
    def update(self, a, b, f):
        g_uot = self.reg*(torch.log(b) - torch.log(torch.exp(f/self.reg)@(self.K)))
        # use f_uot may cause unstable training
        f = self.reg*(torch.log(a) - torch.log(torch.exp(g_uot/self.reg)@(self.K.T)))
        return g_uot, f
    # make prob nozero before dual_value
    def dual_obj_from_f(self, a, b, f):
        g_sink, f_sink = self.update(a, b, f)
        if torch.any(torch.isnan(g_sink)) or torch.any(torch.isnan(f_sink)):
            logger.warn("numerical error nan")
        if torch.any(torch.isinf(g_sink)) or torch.any(torch.isinf(f_sink)):
            logger.warn("numerical error nan")
        g_sink = torch.nan_to_num(g_sink, nan=0.0, posinf=0.0, neginf=0.0)
        f_sink = torch.nan_to_num(f_sink, nan=0.0, posinf=0.0, neginf=0.0)
        return self.dual_value(a,b,f_sink,g_sink),g_sink, f_sink
    def dual_value(self,a,b,f_sink,g_sink):
        dual_obj_left = torch.sum(f_sink * a, dim=-1) + torch.sum(g_sink * b, dim=-1)
        dual_obj_right = - self.reg*torch.sum(torch.exp(f_sink/self.reg)*(torch.exp(g_sink/self.reg)@(self.K.T)), dim = -1)
        dual_obj = dual_obj_left + dual_obj_right
        return dual_obj
    def grad_value(self, a, b, f_sink, g_sink):
        return torch.norm(b - torch.exp(g_sink/self.reg)*(torch.exp(f_sink/self.reg)@(self.K))
                          ,dim = 1,keepdim=True).mean()
    def potential_loss(self, a, b, f_pred, g_pred = None):
        if(g_pred is None):
            g_sink, f_sink = self.update(a, b, f_pred)
        else:
            g_sink = g_pred; f_sink = f_pred
        if torch.any(torch.isnan(g_sink)) or torch.any(torch.isnan(f_sink)):
            logger.warn("numerical error nan")
        if torch.any(torch.isinf(g_sink)) or torch.any(torch.isinf(f_sink)):
            logger.warn("numerical error nan")
        g_sink = torch.nan_to_num(g_sink, nan=0.0, posinf=0.0, neginf=0.0)
        f_sink = torch.nan_to_num(f_sink, nan=0.0, posinf=0.0, neginf=0.0)
        
        loss =  -torch.mean(self.dual_value(a,b,f_sink,g_sink)) 
        return loss
    def consisloss():
        return;
    def dual_loss(self, a, b, f_pred, g_pred = None):
        dual_value,g_s,f_s = self.dual_obj_from_f(a, b, f_pred)
        if dual_value is None:
            return None
        # gradg = b - torch.exp(g_s/self.reg)*(torch.exp(f_s/self.reg)@(self.K))
        # norm2 = torch.norm(gradg,dim = 1,keepdim=True).mean()
        # norm2 = torch.norm(g_s-g_pred ,dim = 1,keepdim=True).mean()
        return - torch.mean(dual_value)
    # make prob nozero before solve
    def solver(self, a,b, max_iter = 1000,ep = 1e-8, init = None,verbose = False, print_interval = 10):
        
        if init is None : 
            f = torch.zeros((a.shape[0],self.K.shape[0]),dtype=torch.float64, device= self.device)
        else:
            f = init
        if verbose:
            start_time = time.time()
            log = {
                "warm_up":init is not None,
                "err":[],
                "quit":0,
            }
        
        for i in range(max_iter):
            g,f = self.update(a,b,f)
            if i % print_interval == 0:
                b0 =  torch.exp(g/self.reg)*(torch.exp(f/self.reg)@(self.K))
                norm2 = torch.norm((b0 - b),dim = 1,keepdim=True)
                if verbose:
                    log["err"].append(torch.sum(norm2).detach().item())
                if i > 0:
                    cond = torch.sum(norm2).item() < ep
                    if  cond:
                        if verbose:
                            log["quit"] = i
                            log["cost"] = time.time() - start_time
                            return g.detach(),f.detach(),log
                        return g.detach(),f.detach()
                    del norm2, cond
        torch.cuda.empty_cache()
        if verbose:
            log["quit"] = max_iter
            log["cost"] = time.time() - start_time
            return g.detach(),f.detach(),log
        return g.detach(),f.detach()
    def P(self,f,g):
        return torch.matmul(
            torch.exp(f.unsqueeze(2)/self.epsilon),
              torch.exp(g.unsqueeze(1)/self.epsilon))*self.K
    
    def otvalue(self,f,g):
        P = self.P(f,g)
        return torch.sum(P*C,dim = (1,2)) + self.epsilon*torch.sum(torch.xlogy(P,P/torch.e),dim = (1,2))
# solver = ot_solver(eps, K, device)
# ##test
# a = torch.abs(torch.rand(batch_size, 1000, device=device))
# b = torch.abs(torch.rand(batch_size, 1000, device=device))
# a = a/a.sum(dim = 1, keepdim = True)
# b = b/b.sum(dim = 1, keepdim = True)
# g,f,l = solver.solver(a, b,verbose=True,max_iter=3000)
# dual_value = solver.dual_value(a,b,f,g)
# otloss = solver.otvalue(f,g)
# pot = solver.potential_loss(a,b,f,g)
# print(l)


# In[7]:


# uot solver, lambda*kl, and pho is same for a and b

class uot_solver():
    def __init__(self, reg, K,device,rho = 1):
        self.reg = self.epsilon  = reg
        self.K = K
        self.device = device
        self.rho = rho
        self.coeff = self.rho*self.epsilon/(self.epsilon+self.rho)
        logging.info("pho : {}".format(self.rho))
    def dualkl(self,y):
        return self.rho*(torch.exp(y/self.rho)- 1)
    
    def update(self, a, b, f):
        g_uot = self.coeff*(torch.log(torch.div(b,torch.exp(f/self.epsilon)@(self.K))))
        f_uot = self.coeff*(torch.log(torch.div(a,torch.exp(g_uot/self.epsilon)@(self.K.T))))
        return g_uot, f_uot   
    # make prob nozero before dual_value
    def dual_obj_from_f(self, a, b, f):
        g_sink, f_sink = self.update(a, b, f)
        if torch.any(torch.isnan(g_sink)) or torch.any(torch.isnan(f_sink)):
            logger.info("numerical error nan")
            return None,None,None
        if torch.any(torch.isinf(g_sink)) or torch.any(torch.isinf(f_sink)):
            logger.info("numerical error nan")
            return None,None,None
        g_sink = torch.nan_to_num(g_sink, nan=0.0, posinf=0.0, neginf=0.0)
        f_sink = torch.nan_to_num(f_sink, nan=0.0, posinf=0.0, neginf=0.0)
        return self.dual_value(a,b,f_sink,g_sink),g_sink, f_sink
    def dual_value(self,a,b,f_sink,g_sink):
        dual_obj_left = - torch.sum(self.dualkl(-f_sink) * a, dim=-1) - torch.sum(self.dualkl(-g_sink) * b, dim=-1)
        dual_obj_right = - self.epsilon*torch.sum(torch.exp(f_sink/self.epsilon)*(torch.exp(g_sink/self.epsilon)@(self.K.T)), dim = -1)
        dual_obj = dual_obj_left + dual_obj_right
        return dual_obj
    def potential_loss(self, a, b, f_pred):
        dual_value,g_s,f_s = self.dual_obj_from_f(a, b, f_pred)
        if dual_value is None:
            return None
        # gradg = b*torch.exp(-g_s/self.rho) - torch.exp(g_s/self.epsilon)*(torch.exp(f_s/self.epsilon)@(self.K))
        # norm2 = torch.norm(gradg,dim = 1,keepdim=True).mean()
        loss =  - torch.mean(dual_value)
        return loss
    # make prob nozero before solve
    def solver(self, a,b, max_iter = 1000,ep = 1e-8, init = None,verbose = False, print_interval = 10):
        if verbose:
            start_time = time.time()
            log = {
                "warm_up":init is not None,
                "err":[],
                "quit":0,
            }
        if init is None : 
            f = torch.zeros((a.shape[0],self.K.shape[0]),dtype=torch.float64, device= self.device)
        else:
            f = init
        
        
        for i in range(max_iter):
            g,f = self.update(a,b,f)
            if i % print_interval == 0:
                norm2 = torch.norm((b*torch.exp(-g/self.rho) - torch.exp(g/self.reg)*(torch.exp(f/self.reg)@(self.K))),dim = 1,keepdim=True)
                if verbose:
                    log["err"].append(torch.sum(norm2).detach().item())
                if i > 0:
                    cond = torch.sum(norm2).item() < ep
                    if  cond:
                        if verbose:
                            log["quit"] = i
                            log["cost"] = time.time() - start_time
                            return g.detach(),f.detach(),log
                        return g.detach(),f.detach()
                    del norm2, cond
        torch.cuda.empty_cache()
        if verbose:
            log["quit"] = max_iter
            log["cost"] = time.time() - start_time

            return g.detach(),f.detach(),log
        return g.detach(),f.detach()
    def P(self,f,g):
        return torch.matmul(
            torch.exp(f.unsqueeze(2)/self.epsilon),
              torch.exp(g.unsqueeze(1)/self.epsilon))*self.K
    
    def otvalue(self,f,g):
        P = self.P(f,g)
        return torch.sum(P*C,dim = (1,2)) + self.epsilon*torch.sum(torch.xlogy(P,P/torch.e),dim = (1,2))
# solver2 = uot_solver(eps, K, device)
# a = torch.abs(torch.rand(batch_size, 1000, device=device))
# b = torch.abs(torch.rand(batch_size, 1000, device=device))
# g,f,l = solver2.solver(a, b,verbose=True)
# dual_value = solver2.dual_value(a,b,f,g)
# otloss = solver2.otvalue(f,g)
# pot = solver2.potential_loss(a,b,f)
# print(l)


# In[8]:


from models import PotentialMLP, DenseICNN,model_


# # OT and UOT loss

# In[9]:


def p_log(l):
    logger.info("warm up:{},err:{:.2f},{:.2f},{:.2f},iter:{},cost:{:.5f}".format(l["warm_up"],l["err"][0],l["err"][1],l["err"][2],l["quit"],l["cost"]))
# wasserstein Loss, OT version
class wloss(torch.nn.Module):
    def __init__(self,reg,K,device):
        super(wloss, self).__init__()
        self.device = device
        self.solver = ot_solver(reg, K, device)
        self.output_size = K.shape[0]
    # a,b is (batch_size,1000),
    # b normed.
    # a,b is not detached.
    def forward(self,a,b,f = None):
        source_unnorm = a.detach()

        source_count = source_unnorm.sum(dim = 1).unsqueeze(1) # sum for all batch, [32]
        source_normed = source_unnorm / source_count # [32,1000]
        g,f,l = self.solver.solver(source_normed, b, init = f,verbose=True)
        p_log(l)
        im_grad_1 = ((source_count) / (source_count * source_count) * f ) # size of [#batch * #cls]
        im_grad_2 = (source_unnorm * f).sum(dim = 1, keepdim = True)/ (source_count * source_count)
        im_grad = (im_grad_1 - im_grad_2).detach()
        # Define loss = <im_grad, predicted density>. The gradient of loss w.r.t prediced density is im_grad.
        loss = torch.sum(a * im_grad)
        return loss
    def potential_loss(self, a, b, f_pred):
        return self.solver.potential_loss(a,b,f_pred)
# a = torch.abs(torch.rand(batch_size, 1000, device=device))
# b = torch.abs(torch.rand(batch_size, 1000, device=device))
# b = b/b.sum(dim = 1, keepdim = True)        
# criterion = wloss(eps, K, device)
# print(criterion(a,b))


# In[29]:
logger.info("no centered")

# like wloss, but input a is viewed as a probability
class wloss_normed(torch.nn.Module):
    def __init__(self,reg,K,device):
        super(wloss_normed, self).__init__()
        self.device = device
        self.solver = ot_solver(reg, K, device)
        self.output_size = K.shape[0]
    # a,b is (batch_size,1000),
    # b normed.
    def forward(self,a,b,f=None):
        g,f,l = self.solver.solver(a, b, init = f,verbose=True)
        loss = (f*a).sum()
        p_log(l)
        return loss,f.detach()
    def potential_loss(self,a,b,f,g = None):
        return self.solver.potential_loss(a,b,f,g_pred= g)
    def grad_loss(self, a,b,f):
        return self.solver.grad_loss(a,b,f,g = None)
    def regress_loss(self,f,real_f):
        return torch.norm(f-real_f,dim = 1,keepdim=True).mean()
# a = torch.abs(torch.rand(batch_size, 1000, device=device))
# b = torch.abs(torch.rand(batch_size, 1000, device=device))
# a = a/a.sum(dim = 1, keepdim = True)
# b = b/b.sum(dim = 1, keepdim = True)        
# criterion = wloss_normed(eps, K, device)
# print(criterion(a,b))


# In[11]:


# UOT loss
class uloss(torch.nn.Module):
    def __init__(self,reg,K,device,rho  = 1):
        super(uloss, self).__init__()
        self.device = device
        self.solver = uot_solver(reg, K, device,rho)
        self.output_size = K.shape[0]
        self.rho = rho
    # a,b is (batch_size,1000) and not normed
    def dualkl(self,y):
        return self.rho*(torch.exp(y/self.rho)- 1)
    def forward(self,a,b,f = None):
        g,f,l = self.solver.solver(a, b, init = f,verbose=True)
        loss = - torch.sum(self.dualkl(-f) * a)
        p_log(l)
        return loss
    def potential_loss(self,a,b,f):
        return self.solver.potential_loss(a,b,f)
a = torch.abs(torch.rand(batch_size, 1000, device=device))
b = torch.abs(torch.rand(batch_size, 1000, device=device))    
criterion = uloss(eps, K, device)
print(criterion(a,b))


# # dataset

# In[12]:


from PIL import Image
import torch.utils.data as data
from glob import glob
import torch,cv2,scipy,os
import torchvision.transforms.functional as F
from torchvision import transforms
import numpy as np
import scipy.io as sio
from torchvision import models as models
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

class ImageDataset(Dataset):
    def __init__(self, mat, method, root_path):
        self.mat = mat
        self.root_path = root_path
        self.image_names = self.mat['flickr_ids'][:2000]
        logger.info('number of img: {} for {}'.format(len(self.image_names),method))

        self.labels = np.array(self.mat['Y'])[:2000]
        self.normed_labels = self.labels + M_EPS
        self.normed_labels = self.normed_labels/self.normed_labels.sum(axis = 1, keepdims = True)
        self.method = method
        # set the training data images and labels
        if self.method == "train":
            # define the training transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((400, 400)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor(),
            ])
        # set the validation data images and labels
        elif self.method == 'val':
            self.image_names = self.image_names[:500]
            self.labels = self.labels[:500]
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((400, 400)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor(),
            ])
            
        elif self.method == 'test':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.root_path, f"{self.image_names[index].strip()}.jpg"))
        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.transform(image)
        targets = self.labels[index]
        normed_targets = self.normed_labels[index]
        
        return {
            'image': image.double(),
            'label': torch.asarray(targets, dtype=torch.float64),
            'normed_label': torch.asarray(normed_targets, dtype=torch.float64),
        }
# test
# read the training mat file
train_mat = scipy.io.loadmat("/home/home/weiyupeng/thome/ym/new_train_label.mat")
val_mat = scipy.io.loadmat("/home/home/weiyupeng/thome/ym/new_val_label.mat")
test_mat = scipy.io.loadmat("/home/home/weiyupeng/thome/ym/new_test_label.mat")
# train dataset
train_data = ImageDataset(
    train_mat, method = "train", root_path = "/home/home/weiyupeng/thome/ym/train"
)
# validation dataset
valid_data = ImageDataset(
    val_mat, method = "val", root_path = "/home/home/weiyupeng/thome/ym/val"
)
# train data loader
train_loader = DataLoader(
    train_data, 
    batch_size=batch_size,
    shuffle=True
)
# validation data loader
valid_loader = DataLoader(
    valid_data, 
    batch_size=batch_size,
    shuffle=False
)


# # train process

# In[13]:


import torch
from tqdm import tqdm
kl_criterion = nn.KLDivLoss(reduction='sum')
alpha = 0.2
opt_interval = args.opt_interval
opt_times = args.opt_times
end_times = args.end_times
logger.info("counter:{}".format(opt_interval))
logger.info("opt times:{}".format(opt_times))
logger.info("stop at {} times".format(end_times))
counter1 = 0
# training function
def train(model, dataloader, optimizer, criterion, train_data, device):
    global pred_optim, pred_net, counter1
    model.train()
    # 选择loss
    choose_loss = 1
    if choose_loss == 1:
        criterion = wloss_normed(eps, K, device)
        logger.info('Training with wloss_normed')
    elif choose_loss == 2:
        criterion = uloss(eps, K, device,rho = 1)
        logger.info('Training with uotloss')
    elif choose_loss == 3:
        criterion = wloss(eps, K, device)
        logger.info('Training with wloss unnormed')
    counter = 0
    train_running_loss = 0.0
    loop = tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size))
    for i, data in loop:
        counter += 1; counter1 += 1
        data, target, labels = data['image'].to(device), data['normed_label'].to(device), data['label']
        loss = torch.tensor(0.0, dtype=torch.float64, device=device)
        optimizer.zero_grad()

        outputs = model(data)
        # apply sigmoid activation to get all the outputs between 0 and 1
        outputs = torch.softmax(outputs,dim = 1)
        outputs_ = outputs.detach()

        a_dens = outputs_.requires_grad_(True)
        criterion.solver.K = K
        # cal opt time
        start_time = time.time()
        ###
        ## wloss_normed, wloss
        b_dens = target.requires_grad_(True)
        ## uloss
        # b_dens = labels
        
        pred_f = pred_net.pred(a_dens, b_dens)
        # 尝试只优化到200次
        if counter1 % opt_interval == 0 and counter1 < end_times:
            for i in range(opt_times):
                ploss = criterion.potential_loss(a_dens, b_dens,pred_f)
                pred_optim.zero_grad()
                ploss.backward()
                pred_optim.step()
                pred_f = pred_net.pred(a_dens, b_dens)
        end_time = time.time()
        logger.info("opt time:{}".format(end_time - start_time))
        # for i in range(labels.size(0)):
        #     clss = labels[i].nonzero()
        #     criterion.solver.K = K[clss,:].squeeze(1).T
        #     target_density = 1/clss.size(0)*torch.ones(1,clss.size(0),device=device)
        #     loss += criterion(outputs[i].unsqueeze(0), target_density,f = pred_f[i].unsqueeze(0))
        #     _ = criterion(outputs[i].unsqueeze(0), target_density)
        # after pass

        #


        temp, real_f = criterion(outputs, target,f = pred_f.detach())
        loss += alpha*temp
        # ploss = criterion.regress_loss(pred_f, real_f)
        # pred_optim.zero_grad()
        # ploss.backward()
        # pred_optim.step()
        loss += kl_criterion(torch.log(outputs), b_dens)

        train_running_loss += loss.item()



        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()
        logging.info("trainloss:{}".format(loss))
        loop.set_postfix(loss=(loss.item()))
        
    train_loss = train_running_loss / counter
    return train_loss


# In[15]:


# validation function
def validate(model, dataloader, criterion, val_data, device):
    print('Validating')
    model.eval()
    counter = 0
    tpk = 0.0
    acc = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            counter += 1
            data, target = data['image'].to(device), data['label'].to(device)
            outputs = model(data)
            # apply sigmoid activation to get all the outputs between 0 and 1
            outputs = torch.softmax(outputs,dim = 1)
            # carry out top K cost
            
            tpk += top_k_cost(2, outputs,target, top_k_matrix)
            acc += multi_label_auc(target, outputs)
            
        
        val_loss = (tpk / counter, acc/ counter)
        return val_loss


# In[30]:


# logging.info("feature this time: add opt time")
# start the training and validation
train_loss = []
valid_loss = []
valid_loss2 = []

model = model_(pretrained=True, requires_grad=False).to(device).double()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
from models import PotentialMLPMean
pred_net = PotentialMLPMean(dim_in=2*1000, dim_out = 1000).to(device).double()

if args.resume :
    pred_net.load_state_dict(torch.load(args.resume, map_location="cpu")['model_state_dict'])
    logger.info("loaded")
pred_net.train()

pred_optim = torch.optim.Adam(pred_net.parameters(), lr = args.lr)
torch.cuda.empty_cache()


# In[ ]:



for epoch in range(epochs):
    logger.info(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, train_loader, optimizer, criterion, train_data, device
    )
    # valid_epoch_loss = validate(
    #     model, valid_loader, criterion, valid_data, device
    # )
    if epoch == 50:
        torch.save({
            'epoch': epoch,
            'model_state_dict': pred_net.state_dict(),
            'optimizer_state_dict': pred_optim.state_dict(),
            'loss': criterion,
            }, './pred_saved/pred_net{}.pth'.format(time_str))
    train_loss.append(train_epoch_loss)
    # valid_loss.append(valid_epoch_loss[0])
    # valid_loss2.append(valid_epoch_loss[1])
    # logger.info(f"Train Loss: {train_epoch_loss:.4f}")
    # logger.info(f'Val Loss: {valid_epoch_loss[0]:.4f},{valid_epoch_loss[1]:.4f}')


# In[ ]:

# save the trained model to disk
torch.save({
            'epoch': epochs,
            'model_state_dict': pred_net.state_dict(),
            'optimizer_state_dict': pred_optim.state_dict(),
            'loss': criterion,
            }, './pred_saved/pred_net{}.pth'.format(time_str))
# plot and save the train and validation line graphs
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(valid_loss, color='red', label='top_k_cost')
plt.plot(valid_loss, color='red', label='acc_curve')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../outputs/loss.png')
plt.show()


# In[ ]:


# 测试：[1,1,1]与[1,M_Eps,....]能差多少，err

# criterion = wloss_normed(eps, K, device)
# model = model_(pretrained=True, requires_grad=False).to(device).double()
# data = next(iter(train_loader))
# data, target,labels = data['image'].to(device), data['normed_label'].to(device),data['label']
# outputs = model(data)
# outputs = torch.sigmoid(outputs)
# outputs = outputs/outputs.sum(dim=1,keepdim=True)
# criterion.solver.K = K
# for i in range(32):
#     criterion(outputs[i].unsqueeze(0), target[i].unsqueeze(0))
# for i in range(32):
#     clss = labels[i].nonzero()
#     criterion.solver.K = K[clss,:].squeeze(1).T

#     criterion(outputs[i].unsqueeze(0), 1/clss.size(0)*torch.ones(1,clss.size(0),device=device))

