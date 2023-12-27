dvc = "0"

from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from munch import Munch
from datetime import datetime
import logging,os

args = {
    "batch_size" : 32,
    "eps" : 0.02,
    "M_EPS" : 1e-16,
    "max_iter" : 1000,
    "err_bound" : 1.,
    "save_root" : "./ckpt",
    "lr" : 0.0001,
    "epochs" : 100,
    "opt_times" : 5,
    "opt_intervals" : 5,
    "opt_totals" : 3000,
    "wanted" : ["resnet", "gauss", "vit", "vgg"],
    "losses" : ["pot", "grad", "reg"],
    "root_dir" : "./generated",
    "train_notes": "val warmups MLP",
    "val_size" : 3000,
    "resume" : "/home/home/weiyupeng/thome/ym/pred_saved/pred_net1211-003605.pth"
    
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
epochs = args.epochs
batch_size = args.batch_size 
eps = args.eps
opt_times = args.opt_times
M_EPS = args.M_EPS
max_iter = args.max_iter
ep = args.err_bound
opt_intervals = args.opt_intervals
opt_totals = args.opt_totals
import scipy
device = torch.device('cuda:{}'.format(str(dvc)) if torch.cuda.is_available() else 'cpu')
C = torch.asarray(scipy.io.loadmat("/home/home/weiyupeng/thome/ym/ground-metric.mat")["M"]).to(device)
K = torch.exp(-C/eps)
import random
class TensorDataset(Dataset):
    def __init__(self, include_list, root_path,val_size) -> None:
        super().__init__()
        paths = [os.path.join(root_path, i) for i in include_list]
        all_tensors_list = []
        for path in paths:
            if(os.path.isdir(path)):
                all_tensors_list += [os.path.join(path, i) for i in os.listdir(path)]            
        all_tensors_list = [x for x in all_tensors_list if "abf-" in x ]
        random.shuffle(all_tensors_list)
        logger.info("num of tensors {}".format(len(all_tensors_list)))
        self.tensor_names = all_tensors_list[:val_size]
    def __len__(self):
        return len(self.tensor_names)
    def __getitem__(self, index):
        ts = torch.load(self.tensor_names[index])
        return ts
# constructor include_list
wanted = args.wanted

def make_list(wanted):
    include_list = []
    if("vgg" in wanted):
        include_list.append("vgg19")
    if("vit" in wanted):
        include_list.append("vit-b16"); include_list.append("vitL-16")
    if("random" in wanted):
        include_list.append("random")
    if("gauss" in wanted):
        include_list.append("gker3"); include_list.append("gker5"); include_list.append("gker7"); 
    if("resnet" in wanted):
        include_list.append("resnet50"); include_list.append("resnet101"); 
    return include_list



from models import PotentialMLPMean, ot_solver, uloss, uot_solver
from tqdm import tqdm
pred_net = PotentialMLPMean(dim_in=2*1000, dim_out = 1000).to(device).double()

if args.resume is not None and args.resume != "":
    pred_net.load_state_dict(torch.load(args.resume,map_location="cpu")['model_state_dict'])
    logger.info("loaded")
pred_opt = torch.optim.Adam(pred_net.parameters(), lr = args.lr)

solver = ot_solver(eps, K, C, logger,device)
def p_log(l):
    if len(l["err"]) < 3:
        logger.info("warm up:{},err:{:.2f},iter:{},cost:{:.5f}".format(l["warm_up"],l["err"][0],l["quit"],l["cost"]))
    else:
        logger.info("warm up:{},err:{:.2f},{:.2f},{:.2f},iter:{},cost:{:.5f}".format(l["warm_up"],l["err"][0],l["err"][1],l["err"][2],l["quit"],l["cost"]))


class PotentialLoss():
    def __init__(self,solver:ot_solver, device):
        self.device = device
        self.solver = solver
        self.output_size = K.shape[0]
    def potential_loss(self,a,b,f,g = None):
        return self.solver.potential_loss(a,b,f,g_pred= g)
    def grad_loss(self, a,b,f):
        return self.solver.grad_loss(a,b,f)
    def regress_loss(self,f,real_f):
        return torch.norm(f-real_f,dim = 1,keepdim=True).mean()
    def forward(self,a,b,f=None):
        g,f,l = self.solver.solver(a, b, init = f,verbose=True)
        loss = (f*a).sum()
        p_log(l)
        return loss,f.detach()
criterion = PotentialLoss(solver, device)

flag_pot = "pot" in args.losses
flag_reg = "reg" in args.losses
flag_grad = "grad" in args.losses
logger.info(" pot {}  reg {}  grad {} ".format(flag_pot,flag_reg, flag_grad))
total_count = 0
pred_net.eval()
import numpy as np

for (include_list, tag) in [(make_list([tag]) , tag) for tag in wanted]:
    try:
        dst = TensorDataset(include_list, args.root_dir, args.val_size)
        train_loader = DataLoader(dst, batch_size=batch_size, shuffle=True)
    except Exception as e:
        logger.info("tag:{} failed")
        continue
    total_zero_err = []; total_f_err = []
    total_zero_iter = []; total_f_iter = []
    
    for i, tensorset in tqdm(enumerate(train_loader), total = int(len(dst)/batch_size)):
        a = tensorset["a"].squeeze(1).to(device).requires_grad_(True)
        b = tensorset["b"].squeeze(1).to(device).requires_grad_(True)
        f = tensorset["f"].squeeze(1).to(device).requires_grad_(True)
        _,_,zero_init  = criterion.solver.solver(a,b,ep = args.err_bound,print_interval=1,verbose=True)
        pred_f = pred_net.pred(a,b)
        
        _,_,f_init = criterion.solver.solver(a,b,init = pred_f,ep = args.err_bound,print_interval=1,verbose=True)
        total_zero_err.append(zero_init["err"][0]); total_f_err.append(f_init["err"][0])
        total_zero_iter.append(zero_init["quit"]); total_f_iter.append(f_init["quit"])
        
        p_log(zero_init); p_log(f_init)
    m1 = np.mean(total_zero_iter)
    m2 = np.mean(total_zero_err)
    m3 = np.mean(total_f_iter)
    m4 = np.mean(total_f_err)
    v1 = np.var(total_zero_iter)
    v2 = np.var(total_zero_err)
    v3 = np.var(total_f_iter)
    v4 = np.var(total_f_err)
    logger.info("Dataset: {},(iter_mean,var, err_mean,var) zero-f:{}-{}|{}-{}|{}-{}|{}-{}".format(tag, m1,v1, m2,v2,m3,v3,m4,v4))    
    
    
    ### train_over
    

torch.save({
        'epoch': epochs,
        'model_state_dict': pred_net.state_dict(),
        'optimizer_state_dict': pred_opt.state_dict(),
        'loss': criterion,
        }, './pred_saved/pred_net{}.pth'.format(time_str))       
        