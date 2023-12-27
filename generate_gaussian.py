# gaussian a vector
from scipy.ndimage import gaussian_filter1d
import numpy as np
import os
from models import ot_solver
# load dataset
import scipy,torch
mat = scipy.io.loadmat("/home/home/weiyupeng/thome/ym/new_train_label.mat")
img_names = mat['flickr_ids'][:4000]
vector = np.array(mat['Y'])[:4000]
print(vector.shape)
generated_path = "./generated/gker"
device = torch.device("cuda:0")
eps = 0.02
C = torch.asarray(scipy.io.loadmat("/home/home/weiyupeng/thome/ym/ground-metric.mat")["M"]).to(device)
K = torch.exp(-C/eps).to(device)
import logging
logger = logging.getLogger()
solver = ot_solver(0.02, K, C,logger,device )
# dataloader
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm

class Dset(Dataset):
    def __init__(self, vector,img_names):
        self.vector = torch.asarray(vector).double()
        self.vector_gaussian_3 = torch.asarray(gaussian_filter1d(vector, 3)).double()
        self.vector_gaussian_5 = torch.asarray(gaussian_filter1d(vector, 5)).double()
        self.vector_gaussian_7 = torch.asarray(gaussian_filter1d(vector, 7)).double()
        self.name_lists = img_names
    def __len__(self):
        return self.vector.shape[0]
    def __getitem__(self, idx):
        return  {
            "b" : self.vector[idx],
            "3": self.vector_gaussian_3[idx],
            "5": self.vector_gaussian_5[idx],
            "7": self.vector_gaussian_7[idx],
            "name" : self.name_lists[idx]
        }
loader = DataLoader(Dset(vector, img_names), batch_size=32, shuffle=True)


for sigma in [3,5,7]:
    generated_sigma_path = generated_path + str(sigma)
    if not os.path.exists(generated_sigma_path):
        os.makedirs(generated_sigma_path)
    
for data in tqdm(loader, total=int(len(loader)/32)):
    b = data["b"].to(device)
    names = data["name"]
    for sigma in ["3","5",'7']:
        generated_sigma_path = generated_path + str(sigma)
        a = data[sigma].to(device)
        _, f= solver.solver(a,b)
        f = f.detach().cpu()
        a = a.detach().cpu()
        b_ = b.detach().cpu()
        for i in range(a.shape[0]):
            torch.save(
                {
                    "a": a[i].unsqueeze_(0),
                    "b": b_[i].unsqueeze_(0), 
                    "f": f[i].unsqueeze_(0) 
                }
                ,generated_sigma_path + "/abf-" + names[i]
            )
        