import torch
import numpy as np
import jax
import jax.numpy as jnp
class Gaussian():
    def __init__(self, mean, cov) -> None:
        dim = mean
def jnptest(points, weights):

    n = points.shape[0]
    if weights is None:
      weights = jnp.ones(n) / n

    mean = weights.dot(points)
    centered_x = (points - mean)
    scaled_centered_x = centered_x * weights.reshape(-1, 1)
    cov = scaled_centered_x.T.dot(centered_x) / (1 - weights.dot(weights))
    return mean,cov
def matrix_powers(m, powers):
    eigs, q = torch.linalg.eigh(m)
    qt = q.permute(0,2,1)
    ret = []
    for power in powers:
        ret.append(torch.matmul(torch.unsqueeze(eigs ** power, -2) * q, qt))
    if len(ret) == 1:
        return ret[0]
    return ret


### geo: [points_number(n), dimension]
### weights : [batch_size, n]
def make_gauss(geo:torch.Tensor, weights:torch.Tensor):
    mean = weights@geo #[batch, dim]
    batch,dim  = weights.shape

    centered_x = geo.repeat(batch,1,1) - mean.unsqueeze(1) #[batch,n,dim]

    scaled_centered_x = centered_x*weights.unsqueeze(-1) #[batch,n,dim]
    cov =  scaled_centered_x.permute(0,2,1)@(centered_x).div((1 - (weights*weights).sum(dim = 1,keepdim=True)).unsqueeze(-1))
    return mean, cov, centered_x

def gmap(cov,cov_dest):
    sqrt0, sqrt0_inv = matrix_powers(cov, (0.5, -0.5))
    sigma1 = cov_dest
    m = torch.asarray(matrix_powers(
        torch.matmul(sqrt0, torch.matmul(sigma1, sqrt0)),(0.5,)
    ))
    return torch.matmul(sqrt0_inv, torch.matmul(m, sqrt0_inv))
class GaInit(torch.nn.Module):
    def __init__(self, geometry_x, geometry_y):
        super().__init__()
        self.geometry_x = geometry_x
        if geometry_y is not None:
            self.geometry_y = geometry_y
        else: self.geometry_y = geometry_x

    def pred(self, a:torch.Tensor,b:torch.Tensor):
        # #### n(1000) is viewed as batch. f is the all n's

        if len(a.shape) == 2: # multi_batch
            batch = a.shape[0]
        # a,b are weight
            x = self.geometry_x
            mean_a, cov_a, centered_x = make_gauss(geo = self.geometry_x, weights= a)
            mean_b, cov_b,_ = make_gauss(geo = self.geometry_y, weights= b)
            a2b = gmap(cov_a,cov_b)
            scaled_x  = torch.bmm(a2b,centered_x.mT) #[batch,dim,n]
            f = 2*(0.5*(x*x).sum(dim=1) - 0.5*(centered_x*scaled_x.mT).sum(dim=-1) - torch.matmul(mean_b,x.T))
            f = f - f.mean(dim = 1, keepdim = True) # this zero-mean f makes the gradient optimize on the simplex
            return  f
        else:
            # batch_size = 1
            pass



# 在一些问题中，几何是很明确的，但是另一些问题中并没有（虽然，可以假定他的几何） 

if __name__ == "__main__":
    p_size_x = 10
    p_size_y = 100
    grid = []
    for i in np.linspace(1, 0, num = p_size_x):
        for j in np.linspace(0, 1, num = p_size_y):
            grid.append([j, i])
    x_grid = torch.asarray(grid)
    y_grid = torch.asarray(grid)
    ginit = GaInit(x_grid,y_grid)
    ba = 20
    a = torch.abs(torch.randn((ba,1000),dtype = torch.float64))
    a = a/torch.sum(a,dim=1,keepdim = True)
    b = torch.abs(torch.randn((ba,1000),dtype = torch.float64))
    b = b/torch.sum(b,dim=1,keepdim = True)
    # m = ginit.pred(a = a,
    #            b = a)
    f =ginit.pred(a = a,b=b)
    
    from ott.geometry.pointcloud import PointCloud
    from ott.problems.linear.linear_problem import LinearProblem
    from ott.initializers.linear.initializers import GaussianInitializer
    # GT
    # this code only saves one batch each time
    for i in range(ba):

        geom = PointCloud(x = jnp.array(x_grid), y = jnp.array(x_grid), epsilon = 0.05)

        gi = GaussianInitializer()
        p = LinearProblem(geom=geom,a=jnp.array(a[i]),b = jnp.array(b[i]))
        f1 = gi.init_dual_a(p,lse_mode=True)
        print(torch.norm(torch.abs(f[i]-torch.asarray(f1))))
