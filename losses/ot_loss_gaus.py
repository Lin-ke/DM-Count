import torch,time,logging
from torch.nn import Module
from models import MLP

from utils.log_utils import get_value
from .bregman_pytorch import sinkhorn
from gaus import GaInit
import numpy as np
M_EPS = 1e-16
from munch import Munch
import os
class OT_Loss(Module):
    def __init__(self, c_size, stride, norm_cood, device, num_of_iter_in_ot=100, reg=10.0):
        super(OT_Loss, self).__init__()
        assert c_size % stride == 0

        self.c_size = c_size
        self.device = device
        self.norm_cood = norm_cood
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg

        # coordinate is same to image space, set to constant since crop size is same
        self.cood = torch.arange(0, c_size, step=stride,
                                 dtype=torch.float32, device=device) + stride / 2
        self.density_size = self.cood.size(0)
        self.cood.unsqueeze_(0) # [1, #cood]
        if self.norm_cood:
            self.cood = self.cood / c_size * 2 - 1 # map to [-1, 1]
        self.output_size = self.cood.size(1)

        shape = self.c_size //stride #(64)
        
        self.pred_net = MLP(2*shape**2, shape**2).to(self.device) # (input:shape**2(est count) + shape**2(gt count), output:shape(vector a))
        self.pred_optim = torch.optim.Adam(self.pred_net.parameters(), lr = 1e-3)
        self.logger = get_value("Logger")
        self.predcounter = 0
        self.schedualcounter = 0
        p_size_x = shape
        p_size_y = shape
        grid = []
        for i in np.linspace(1, 0, num = p_size_x):
            for j in np.linspace(0, 1, num = p_size_y):
                grid.append([j, i])
        x_grid = torch.asarray(grid,device = device,dtype = torch.float32)
        y_grid = torch.asarray(grid, device=device,dtype = torch.float32)
        self.gaus = GaInit(x_grid, y_grid)
        args = {
            "train notes" : "1. method one by one, train pred network. 2. add gauss",
            "opt_interval" : 1,
            "opt_times" : 10,
            "end_times" : 100000,
            "resume":None,
            "save_dir" : get_value("save_dir"),
            "start_use_init" : 0,
            
        }
        
        args = Munch(args)        
        self.args = args
        os.system("cp {} {}".format(os.path.realpath(__file__), args.save_dir))
        self.logger.info("cp {} {}".format(os.path.realpath(__file__), args.save_dir))
        for k, v in args.items():
            self.logger.info("{}:\t{}".format(k.ljust(15), v))
        self.counter = 0
        
    def print_log(self, log, descriptor = "default"):
        logger = self.logger

        errs = log["err"]
        if len(errs) >= 3:
            logger.info("{},,{},,{},,{},,{}".format(descriptor,errs[0],errs[1],errs[2],log["it"]))
        else: logger.info("{},,{},,{},,{},,{}".format(descriptor,errs[0],0,0,log["it"]))
        
      
    def forward(self, normed_density, unnormed_density, points, gt_discrete):
        args=  self.args
        opt_interval = self.args.opt_interval
        opt_times = self.args.opt_interval
        end_times = self.args.end_times
        start_use_init = self.args.start_use_init
        logger = self.logger
        
        batch_size = normed_density.size(0)
        assert len(points) == batch_size
        assert self.output_size == normed_density.size(2)
        loss = torch.zeros([1]).to(self.device)
        dual_loss = torch.zeros([1]).to(self.device)
        ot_obj_values = torch.zeros([1]).to(self.device)
        wd = 0 # wasserstain distance
        for idx, im_points in enumerate(points):
            if len(im_points) >= 1:
                self.schedualcounter += 1
                # compute l2 square distance, it should be source target distance. [#gt, #cood * #cood]
                if self.norm_cood:
                    im_points = im_points / self.c_size * 2 - 1 # map to [-1, 1]
                x = im_points[:, 0].unsqueeze_(1)  # [#gt, 1]
                y = im_points[:, 1].unsqueeze_(1)
                x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood # [#gt, #cood]
                y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood
                y_dis.unsqueeze_(2)
                x_dis.unsqueeze_(1)
                dis = y_dis + x_dis
                dis = dis.view((dis.size(0), -1)) # size of [#gt, #cood * #cood]
                
                
                source_prob = normed_density[idx][0].view([-1]).detach()
                target_prob = (torch.ones([len(im_points)]) / len(im_points)).to(self.device)
                self.K = torch.exp(torch.div(dis, -self.reg)).T
                # use sinkhorn to solve OT, compute optimal beta.
                # *warm start: f_pred
                
                
                source_prob_ = source_prob.unsqueeze(0)
                
                # Add, 并不优化
                start_time = time.time()
                if self.schedualcounter % opt_interval == 0 and self.schedualcounter < end_times and self.schedualcounter >= start_use_init:
                    for i in range(opt_times):
                        G_pred = self.pred_net(source_prob_,gt_discrete[idx].reshape(1,-1))
                        pred_loss = self.potential_loss(a =source_prob,b = target_prob, f_pred = G_pred)
                        if pred_loss: # nan时跳过
                            self.pred_optim.zero_grad()
                            pred_loss.backward()
                            self.pred_optim.step()

                end_time = time.time()
                # self.logger.info("optmize time: {}".format(end_time-start_time))
                
                ### Gaus
                gaus_pred = self.gaus.pred(source_prob_,gt_discrete[idx].reshape(1,-1))
                ###

                # 测试不使用sinkhorn，优化5次,用switch切换
                # 测试不使用sinkhorn，加损失
                # 推理 + 计算
                switch = self.schedualcounter > start_use_init
                if self.schedualcounter == start_use_init:
                    logger.info("start using init")
                if switch:
                    start_time = time.time()
                    G_pred = self.pred_net(source_prob_,gt_discrete[idx].reshape(1,-1))
                    P, log = sinkhorn(target_prob, source_prob, dis, self.reg,warm_start=None, maxIter=self.num_of_iter_in_ot, log=True)
                    self.print_log(log, "base_line {}".format(self.schedualcounter))
                    P, log = sinkhorn(target_prob, source_prob, dis, self.reg,warm_start=gaus_pred.squeeze(0).detach(), maxIter=self.num_of_iter_in_ot, log=True)
                    self.print_log(log, "gaus_pred")

                    P, log = sinkhorn(target_prob, source_prob, dis, self.reg,warm_start=G_pred.squeeze(0).detach(), maxIter=self.num_of_iter_in_ot, log=True)
                    
                    self.print_log(log, "net_pred")
                    end_time = time.time()
                    # self.logger.info("emd time with init: {}".format(end_time-start_time))
                    # self.logger.info(log['err'][0])
                    # self.logger.info(log['it'])
                    beta = G_pred.squeeze(0).detach()
                
                
                # 直接Sinkhorn
                else:
                    start_time = time.time()
                    P, log = sinkhorn(target_prob, source_prob, dis, self.reg,warm_start=None, maxIter=self.num_of_iter_in_ot, log=True)
                    end_time = time.time()
                    # self.logger.info("emd time without init: {}".format(end_time-start_time))
                    # self.logger.info(log['err'][0])
                    # self.logger.info(log['it'])
                    self.print_log(log, "base_line {}".format(self.schedualcounter))
                    beta = log['beta'] # size is the same as source_prob: [#cood * #cood]
                ot_obj_values += torch.sum(normed_density[idx] * beta.view([1, self.output_size, self.output_size]))
                # compute the gradient of OT loss to predicted density (unnormed_density).
                # im_grad = beta / source_count - < beta, source_density> / (source_count)^2
                source_density = unnormed_density[idx][0].view([-1]).detach()
                source_count = source_density.sum()
                im_grad_1 = (source_count) / (source_count * source_count+1e-8) * beta # size of [#cood * #cood]
                im_grad_2 = (source_density * beta).sum() / (source_count * source_count + 1e-8) # size of 1
                im_grad = im_grad_1 - im_grad_2
                im_grad = im_grad.detach().view([1, self.output_size, self.output_size])
                # Define loss = <im_grad, predicted density>. The gradient of loss w.r.t prediced density is im_grad.
                loss += torch.sum(unnormed_density[idx] * im_grad)
                wd += torch.sum(dis * P).item()

        return loss, wd, ot_obj_values

    def update(self, a, b, f):
        g_uot = self.reg*(torch.log(b) - torch.log(torch.exp(f/self.reg)@(self.K) + M_EPS))
        # use f_uot may cause unstable training
        f = self.reg*(torch.log(a) - torch.log(torch.exp(g_uot/self.reg)@(self.K.T)+M_EPS))
        return g_uot, f
    def dual_obj_from_f(self, a, b, f):
        g_sink, f_sink = self.update(a, b, f)
        if torch.any(torch.isnan(g_sink)) or torch.any(torch.isnan(f_sink)):
            self.logger.info("numerical error nan")
            return None,None,None
        if torch.any(torch.isinf(g_sink)) or torch.any(torch.isinf(f_sink)):
                self.logger.info("numerical error nan")
                return None,None,None
        g_sink = torch.nan_to_num(g_sink, nan=0.0, posinf=0.0, neginf=0.0)
        f_sink = torch.nan_to_num(f_sink, nan=0.0, posinf=0.0, neginf=0.0)
        dual_obj_left = torch.sum(f_sink * a, dim=-1) + torch.sum(g_sink * b, dim=-1)
        dual_obj_right = - self.reg*torch.sum(torch.exp(f_sink/self.reg)*(torch.exp(g_sink/self.reg)@(self.K.T)), dim = -1)
        dual_obj = dual_obj_left + dual_obj_right
        return dual_obj, g_sink, f_sink
    def potential_loss(self, a, b, f_pred):
        dual_value,g_s,f_s = self.dual_obj_from_f(a+M_EPS, b+M_EPS, f_pred)
        if dual_value is None:
            return None
        # gradg = b - torch.exp(g_s/self.reg)*(torch.exp(f_s/self.reg)@(self.K))
        # norm2 = torch.norm(gradg,dim = 1,keepdim=True).mean()
        loss =  - torch.mean(dual_value)
        return loss