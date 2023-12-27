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
    "err_bound" : 1e-8,
    "save_root" : "./ckpt",
    "lr" : 0.0001,
    "epochs" : 100,
    "opt_times" : 10,
    "wanted" : ["vgg", "resnet",  "vit"],
    "losses" : ["pot", "grad", "reg"],
    "root_dir" : "./generated",
    "train_notes": "train warmups MLP",
    "resume" : "/home/home/weiyupeng/thome/ym/pred_saved/pred_net1211-003605.pth",
    "train_size" : 4000
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
import scipy
device = torch.device('cuda:{}'.format(str(dvc)) if torch.cuda.is_available() else 'cpu')
C = torch.asarray(scipy.io.loadmat("/home/home/weiyupeng/thome/ym/ground-metric.mat")["M"]).to(device)
K = torch.exp(-C/eps)
import random
class TensorDataset(Dataset):
    def __init__(self, include_list, root_path, train_size = args.train_size) -> None:
        super().__init__()
        paths = [os.path.join(root_path, i) for i in include_list]
        all_tensors_list = []
        for path in paths:
            if(os.path.isdir(path)):
                all_tensors_list += [os.path.join(path, i) for i in os.listdir(path)]            
        all_tensors_list = [x for x in all_tensors_list if "abf-" in x ]
        random.shuffle(all_tensors_list)
        logger.info("num of tensors {}".format(len(all_tensors_list)))
        self.tensor_names = all_tensors_list[:train_size]
    def __len__(self):
        return len(self.tensor_names)
    def __getitem__(self, index):
        ts = torch.load(self.tensor_names[index])
        return ts
# constructor include_list
include_list = []
wanted = args.wanted
if("vgg" in wanted):
    include_list.append("vgg19")
if("vit" in wanted):
    include_list.append("vit-b16"); include_list.append("vitL-16")
if("random" in wanted):
    include_list.append("random")
if("gauss" in wanted):
    include_list.append("gker3");include_list.append("gker5");include_list.append("gker7"); 
if("resnet" in wanted):
    include_list.append("resnet50"); include_list.append("resnet101"); 


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

from models import EMA

ema = EMA(model=pred_net, decay=0.99)


for epoch in range(epochs):
    if epoch % 10 == 0 and epoch > 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': pred_net.state_dict(),
            'optimizer_state_dict': pred_opt.state_dict(),
            'loss': criterion,
            }, './pred_saved/pred_net{}.pth'.format(time_str))
    ### train_process
    dst = TensorDataset(include_list, args.root_dir)
    train_loader = DataLoader(dst, batch_size=batch_size, shuffle=True)

    for i, tensorset in tqdm(enumerate(train_loader), total = int(len(dst)/batch_size)):
        a = tensorset["a"].squeeze(1).to(device).requires_grad_(True)
        b = tensorset["b"].squeeze(1).to(device).requires_grad_(True)
        f = tensorset["f"].squeeze(1).to(device).requires_grad_(True)
        for opt_times in range(opt_times):
            ploss = torch.tensor(0.0, dtype=torch.float64, device=device)
            pred_opt.zero_grad()
            pred_f = pred_net.pred(a,b)
            if flag_pot:
                ploss += criterion.potential_loss(a,b,pred_f)
            if flag_grad:
                ploss += criterion.grad_loss(a,b,pred_f)   
            ploss.backward()
            pred_opt.step()
        
            
        ## out for
        if flag_reg:
            pred_opt.zero_grad()
            pred_f = pred_net.pred(a,b)
            ploss = criterion.regress_loss(f,pred_f)
            ploss.backward()
            pred_opt.step()
        # 查看效果
        # temp, real_f = criterion.forward(a, b,f = None)
        # temp, real_f = criterion.forward(a, b,f = pred_f)
        logger.info( "False {}".format(criterion.solver.initvalue(a,b,init = None)))
        logger.info( "True {}".format(criterion.solver.initvalue(a,b,init = pred_f)))
        
        
        
    
    ### train_over
    

torch.save({
        'epoch': epochs,
        'model_state_dict': pred_net.state_dict(),
        'optimizer_state_dict': pred_opt.state_dict(),
        'loss': criterion,
        }, './pred_saved/pred_net{}.pth'.format(time_str))       
        