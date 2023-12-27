# prednet
from torchvision import models as models
import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

__all__ = ['vgg19']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',

}

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    def forward(self, x):
        x = self.features(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer(x)
        mu = self.density_layer(x)
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}


def vgg19():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model
def vgg16():
    model = VGG(make_layers(cfg['D']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)
    return model
    


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_num = 4, hidden_dim = 1024):
        super().__init__()
        self.model_base = nn.ModuleList([
            nn.ModuleList([nn.Linear(dim_in, hidden_dim, dtype=torch.float32) if i == 0 else nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.ReLU()])
            for i in range(hidden_num)
        ])
        self.lin = nn.Linear(hidden_dim, dim_out, dtype=torch.float32)

    def forward(self, a, b):
        x = torch.cat((a, b), dim = -1)
        for L, A in self.model_base:
            x = A(L(x))
        return self.lin(x)

    
def model_(pretrained = True, requires_grad = True):
    model = models.resnet101(progress=True, weights=models.ResNet101_Weights.DEFAULT)
    # to freeze the hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    # to train the hidden layers
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    # make the classification layer learnable
    # we have 25 classes in total
    model.fc = nn.Linear(2048, 4096)
    return model
class resnet50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model_()
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.model(x)
        x = self.act(x)
        outputs_normed = x/(x+ 1e-6).sum(dim=1,keepdim=True)
        return x,outputs_normed
class vitb16(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model_vit()
        self.act = nn.ReLU()
        self.head = nn.Linear(1000, 4096)
    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        x = self.act(x)
        outputs_normed = x/(x+ 1e-6).sum(dim=1,keepdim=True)
        return x,outputs_normed
class vitl16(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model_vitl()
        self.act = nn.ReLU()
        self.head = nn.Linear(1000, 4096)
    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        x = self.act(x)
        outputs_normed = x/(x+ 1e-6).sum(dim=1,keepdim=True)
        return x,outputs_normed
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original_params = {}
        self.model = model

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original_params[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow and name in self.original_params
                param.data = self.original_params[name]

def model_vit(pretrained =  True, requires_grad = True):
    model = models.vit_b_16(progress=True, weights=models.ViT_B_16_Weights.DEFAULT)
    # to freeze the hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    return model
def model_vitl(pretrained = True, requires_grad = True):
    model = models.vit_l_16(progress=True, weights=models.ViT_L_16_Weights.DEFAULT)
    # to freeze the hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    # to train the hidden layers
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    # make the classification layer learnable
    # we have 25 classes in total
    return model
class PotentialMLP(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_num = 3,act = nn.ReLU(), hidden_dim = 1024):
        super().__init__()
        self.model_base = nn.ModuleList([
            nn.ModuleList([nn.Linear(dim_in, hidden_dim, dtype=torch.float64) if i == 0 else nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64),
            act])
            for i in range(hidden_num)
        ])
        self.lin = nn.Linear(hidden_dim, dim_out, dtype=torch.float64)

    def forward(self, a, b):
        x = torch.cat((a, b), dim = -1)
        for L, A in self.model_base:
            x = A(L(x))
        return self.lin(x)
    def pred(self, a, b):
        return self.forward(a,b)
class PotentialMLPMean(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_num = 3,act = nn.ReLU(), hidden_dim = 1024):
        super().__init__()
        self.model_base = nn.ModuleList([
            nn.ModuleList([nn.Linear(dim_in, hidden_dim, dtype=torch.float64) if i == 0 else nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64),
            act])
            for i in range(hidden_num)
        ])
        self.lin = nn.Linear(hidden_dim, dim_out, dtype=torch.float64)

    def forward(self, a, b):
        x = torch.cat((a, b), dim = -1)
        for L, A in self.model_base:
            x = A(L(x))
        return self.lin(x)
    def pred(self, a, b):
        f = self.forward(a,b)
        return f - f.mean(dim= 1, keepdim = True)
class PotentialSkipMLP(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_num = 8,act = nn.ReLU(), hidden_dim = 256, skip_connections = (4, ), out_act = None):
        super().__init__()
        self.skip_connections = skip_connections
        self._skip_connections= set(skip_connections) if skip_connections else set()
        layers = []
        for i in range(hidden_num):
            model_base = []
            if i == 0:
                model_base.append(nn.Linear(dim_in, hidden_dim, dtype=torch.float64))
            elif i in self._skip_connections:
                model_base.append(nn.Linear(hidden_dim+dim_in , hidden_dim))
            else: model_base.append(nn.Linear(hidden_dim, hidden_dim))
            model_base.append(act)
            layers.append(nn.ModuleList(model_base))
    
        self.model_base = nn.ModuleList(layers)
        self.lin = nn.Linear(hidden_dim, dim_out, dtype=torch.float64)
        self.out_act = out_act # not used
    # @TODO: make another dimension to encode some other informations.
        
    def forward(self, a, b):
        x = torch.cat((a, b), dim = -1)
        _input = x
        for i, (L, A) in enumerate(self.model_base):
            if i in self._skip_connections:
                x = A(L(torch.cat((_input, x), -1)))
            else:
                x = A(L(x))
        return self.lin(x)
    def pred(self, a, b):
        return self.forward(a,b)
class DenseICNN(nn.Module):
    '''Fully Conncted ICNN with input-quadratic skip connections'''
    def __init__(
        self, in_dim, 
        hidden_layer_sizes=[32, 32, 32],
        rank=1, activation='celu', 
        strong_convexity=1e-6,device = 'cuda:0'
    ):
        super(DenseICNN, self).__init__()
        self.device = device
        self.strong_convexity = strong_convexity
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.rank = rank
        
        self.quadratic_layers = nn.ModuleList([
            nn.Sequential(
                ConvexQuadratic(in_dim, out_features, rank=rank, bias=True),
            )
            for out_features in hidden_layer_sizes
        ])
        
        sizes = zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:])
        self.convex_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
            )
            for (in_features, out_features) in sizes
        ])
        if self.activation == 'celu':
            self.act = torch.nn.CELU()
        elif self.activation == 'softplus':
            self.act = torch.nn.Softplus()
        else:
            raise Exception('Activation is not specified or unknown.')

        self.final_layer = nn.Linear(hidden_layer_sizes[-1], 1, bias=False)

    def forward(self, input_a, input_b):
        input = torch.cat((input_a, input_b), dim = -1)
        output = self.quadratic_layers[0](input)
        for quadratic_layer, convex_layer in zip(self.quadratic_layers[1:], self.convex_layers):
            output = convex_layer(output) + quadratic_layer(input)
            output = self.act(output)
        return self.final_layer(output) + .5 * self.strong_convexity * (input ** 2).sum(dim=1).reshape(-1, 1)
    
    def push(self, input_a, input_b):
        output = autograd.grad(
            outputs=self.forward(input_a,input_b), inputs=input_a,
            create_graph=True, retain_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((input_a.size()[0], 1)).to(device=self.device).float()
        )[0]
        return output    
    def pred(self, input_a,input_b):
        return self.push(input_a,input_b)
    def convexify(self):
        for layer in self.convex_layers:
            for sublayer in layer:
                if (isinstance(sublayer, nn.Linear)):
                    sublayer.weight.data.clamp_(0)

        self.final_layer.weight.data.clamp_(0)



class ConvexQuadratic(nn.Module):
    '''Convex Quadratic Layer'''
    __constants__ = ['in_features', 'out_features', 'quadratic_decomposed', 'weight', 'bias']

    def __init__(self, in_features, out_features, bias=True, rank=1):
        super(ConvexQuadratic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        self.quadratic_decomposed = nn.Parameter(torch.Tensor(
            torch.randn(in_features, rank, out_features)
        ))
        self.weight = nn.Parameter(torch.Tensor(
            torch.randn(out_features, in_features)
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        quad = ((input.matmul(self.quadratic_decomposed.transpose(1,0)).transpose(1, 0)) ** 2).sum(dim=1)
        linear = F.linear(input, self.weight, self.bias)
        return quad + linear
# prednet
from torchvision import models as models
import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.nn.functional as F
class DoubleICNN(nn.Module):
    '''Fully Conncted ICNN with input-quadratic skip connections'''
    def __init__(
        self, in_dim:tuple, 
        hidden_layer_sizes_a=[32, 32, 32],
        hidden_layer_sizes_b=[32, 32, 32],
        rank=1, activation='celu', 
        strong_convexity=1e-6,device = 'cuda:0'
    ):
        super(DoubleICNN, self).__init__()
        self.device = device
        self.strong_convexity = strong_convexity
        self.activation = activation
        self.rank = rank
        self.icnna = DenseICNN(in_dim[0], hidden_layer_sizes_a, rank, activation, strong_convexity, device)
        self.icnnb = DenseICNN(in_dim[1], hidden_layer_sizes_b, rank, activation,  strong_convexity, device)
        self.a_path = self.icnna.convex_layers
        self.b_path = self.icnnb.convex_layers
        self.a_a_path = self.icnna.quadratic_layers
        self.b_b_path = self.icnnb.quadratic_layers
        size_a = [in_dim[0]] + hidden_layer_sizes_a[:-1]
        size_b = [in_dim[1]] + hidden_layer_sizes_b[:-1]
        self.length = len(hidden_layer_sizes_a)
        self.a_b_path = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_features, bias=True),
            )
            for in_features,out_features in zip(size_a, size_b)
        ])
        self.b_a_path = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_features, bias=True),
            )
            for in_features,out_features in zip(size_b, size_a)
        ])
        print('a_a_path',self.a_a_path)
        print('b_b_path',self.b_b_path)
        print('a_b_path',self.a_b_path)
        print('b_a_path',self.b_a_path)
        if self.activation == 'celu':
            self.act = torch.nn.CELU()
        elif self.activation == 'softplus':
            self.act = torch.nn.Softplus()
        else:
            raise Exception('Activation is not specified or unknown.')

    def forward(self, a,b):
        z = self.b_b_path[0](b * (F.relu(self.a_b_path[0](a))))
        z = self.act(z)
        u = self.a_a_path[0](a * (F.relu(self.b_a_path[0](b))))
        u = self.act(u)
        for i in range(self.length):
            u_1 = self.a_path[i](u * F.relu(self.b_a_path[i + 1](u))) + self.a_a_path[i + 1](a)
            u_1 = self.act(u_1)
            z_1 = self.b_path[i](z * F.relu(self.a_b_path[i + 1](z))) + self.b_b_path[i + 1](b)
            z_1 = self.act(z_1)
            u = u_1
            z = z_1

        return (self.icnna.final_layer(u) + self.icnnb.final_layer(z))/100
    
    def push(self, input):
        output = autograd.grad(
            outputs=self.forward(input), inputs=input,
            create_graph=True, retain_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((input.size()[0], 1)).to(device=self.device).float()
        )[0]
        return output    
    
    def convexify(self):
        for icnn in [self.icnn_a, self.icnn_b]:
            icnn.convexify()
class OriDenseICNN(nn.Module):
    '''Fully Conncted ICNN with input-quadratic skip connections'''
    def __init__(
        self, in_dim, 
        hidden_layer_sizes=[32, 32, 32],
        rank=1, activation='celu', 
        strong_convexity=1e-2,device = 'cuda:0'
    ):
        super(OriDenseICNN, self).__init__()
        self.device = device
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.rank = rank
        self.strong_convexity = strong_convexity
        
        self.quadratic_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_features, bias=True),
            )
            for out_features in self.hidden_layer_sizes
        ])
        
        sizes = zip(self.hidden_layer_sizes[:-1], self.hidden_layer_sizes[1:])
        self.convex_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
            )
            for (in_features, out_features) in sizes
        ])
        self.half_input =int( in_dim /2); 
        if self.activation == 'celu':
            self.act = torch.nn.CELU()
        elif self.activation == 'softplus':
            self.act = torch.nn.Softplus()
        elif self.activation == 'elu':
            self.act = torch.nn.ELU()
        elif self.activation == "leakyrelu":
            self.act = torch.nn.LeakyReLU()
        else:
            raise Exception('Activation is not specified or unknown.')
        self.final_layer = nn.Linear(hidden_layer_sizes[-1], 1, bias=False)

    # def forward(self, input_a, input_b):
    #     input = torch.cat((input_a, input_b), dim = -1)
    #     output = self.quadratic_layers[0](input)
    #     for quadratic_layer, convex_layer in zip(self.quadratic_layers[1:], self.convex_layers):
    #         output = convex_layer(output) + quadratic_layer(input)
    #         output = self.act(output)
    #     return torch.abs(self.final_layer(output) )
    def forward(self, input_a, input_b):
        input = torch.cat((input_a, input_b), dim = -1)
        output = self.quadratic_layers[0](input)
        for quadratic_layer, convex_layer in zip(self.quadratic_layers[1:], self.convex_layers):
            output = convex_layer(output) + quadratic_layer(input)
            output = self.act(output)
        return self.final_layer(output) + .5 * self.strong_convexity * (input ** 2).sum(dim=1).reshape(-1, 1)
    def push(self, input_a, input_b):
        output = autograd.grad(
            outputs=self.forward(input_a,input_b), inputs=input_a,
            create_graph=True, retain_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((input_a.size()[0], 1)).to(device=self.device).float()
        )[0]
        return output    
    def pred(self, input_a,input_b):
        f = self.push(input_a,input_b)
        return f - f.mean(dim=1, keepdim = True)
    def convexify(self):
        for layer in self.convex_layers:
            for sublayer in layer:
                if (isinstance(sublayer, nn.Linear)):
                    sublayer.weight.data.clamp_(0)

        self.final_layer.weight.data.clamp_(0)


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.position_encoding = PositionalEncoding(hidden_dim)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.position_encoding(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_length=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_length, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)
import math
class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(TransformerLayer, self).__init__()
        self.multihead_attention = MultiheadAttention(hidden_dim, num_heads)
        self.feed_forward = FeedForward(hidden_dim)

    def forward(self, x):
        x = self.multihead_attention(x)
        x = self.feed_forward(x)
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size = x.size(0)

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)

        x = torch.matmul(attention_weights, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)

        x = self.fc(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class NewDenseICNN(nn.Module):
    '''Fully Conncted ICNN with input-quadratic skip connections'''
    def __init__(
        self, in_dim, 
        hidden_layer_sizes=[32, 32, 32],
        rank=1, activation='celu', 
        strong_convexity=1e-2,device = 'cuda:0'
    ):
        super(NewDenseICNN, self).__init__()
        self.device = device
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.rank = rank
        self.strong_convexity = strong_convexity
        
        self.quadratic_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_features, bias=True),
            )
            for out_features in self.hidden_layer_sizes
        ])
        
        sizes = zip(self.hidden_layer_sizes[:-1], self.hidden_layer_sizes[1:])
        self.convex_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
            )
            for (in_features, out_features) in sizes
        ])
        self.half_input =int( in_dim /2); 
        if self.activation == 'celu':
            self.act = torch.nn.CELU()
        elif self.activation == 'softplus':
            self.act = torch.nn.Softplus()
        elif self.activation == 'elu':
            self.act = torch.nn.ELU()
        elif self.activation == "leakyrelu":
            self.act = torch.nn.LeakyReLU()
        else:
            raise Exception('Activation is not specified or unknown.')
        self.final_layer = nn.Linear(hidden_layer_sizes[-1], 1, bias=False)

    # def forward(self, input_a, input_b):
    #     input = torch.cat((input_a, input_b), dim = -1)
    #     output = self.quadratic_layers[0](input)
    #     for quadratic_layer, convex_layer in zip(self.quadratic_layers[1:], self.convex_layers):
    #         output = convex_layer(output) + quadratic_layer(input)
    #         output = self.act(output)
    #     return torch.abs(self.final_layer(output) )
    def forward(self, input_a, input_b):
        input = input_a
        output = self.quadratic_layers[0](input)
        for quadratic_layer, convex_layer in zip(self.quadratic_layers[1:], self.convex_layers):
            output = convex_layer(output) + quadratic_layer(input_b)
            output = self.act(output)
        return self.final_layer(output) + .5 * self.strong_convexity * (input ** 2).sum(dim=1).reshape(-1, 1)
    def push(self, input_a, input_b):
        output = autograd.grad(
            outputs=self.forward(input_a,input_b), inputs=input_a,
            create_graph=True, retain_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((input_a.size()[0], 1)).to(device=self.device).float()
        )[0]
        return output    
    def pred(self, input_a,input_b):
        return self.push(input_a,input_b)
    def convexify(self):
        for layer in self.convex_layers:
            for sublayer in layer:
                if (isinstance(sublayer, nn.Linear)):
                    sublayer.weight.data.clamp_(0)

        self.final_layer.weight.data.clamp_(0)
class MirrorICNN(nn.Module):
    def __init__(
        self, in_dim, 
        hidden_layer_sizes=[32, 32, 32],
        rank=1, activation='celu', 
        strong_convexity=1e-6,device = 'cuda:0'
    ):
        super(MirrorICNN, self).__init__()
        self.device = device
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.rank = rank
        self.strong_convexity = strong_convexity
        
        self.quadratic_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_features, bias=True),
            )
            for out_features in self.hidden_layer_sizes
        ])
        
        sizes = zip(self.hidden_layer_sizes[:-1], self.hidden_layer_sizes[1:])
        self.convex_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
            )
            for (in_features, out_features) in sizes
        ])
        self.half_input =int( in_dim /2); 
        if self.activation == 'celu':
            self.act = torch.nn.CELU()
        elif self.activation == 'softplus':
            self.act = torch.nn.Softplus()

        elif self.activation == "leakyrelu":
            self.act = torch.nn.LeakyReLU()
        else:
            raise Exception('Activation is not specified or unknown.')
        self.final_layer = nn.Linear(hidden_layer_sizes[-1], 1, bias=False)

    def forward(self, input_a, input_b):
        input = torch.cat((input_a, input_b), dim = -1)
        output = self.quadratic_layers[0](input)
        for quadratic_layer, convex_layer in zip(self.quadratic_layers[1:], self.convex_layers):
            output = convex_layer(output) + quadratic_layer(input)
            output = self.act(output)
        return self.final_layer(output) + .5 * self.strong_convexity * (input ** 2).sum(dim=1).reshape(-1, 1)
    
    def push(self, input_a, input_b):
        output = autograd.grad(
            outputs=self.forward(input_a,input_b), inputs=input_a,
            create_graph=True, retain_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((input_a.size()[0], 1)).to(device=self.device).float()
        )[0]
        return output    
    def pred(self, input_a,input_b):
        return self.push(input_a,input_b)
    def convexify(self):
        for layer in self.convex_layers:
            for sublayer in layer:
                if (isinstance(sublayer, nn.Linear)):
                    sublayer.weight.data.clamp_(0)

        self.final_layer.weight.data.clamp_(0)

class DynamicDenseICNN(nn.Module):
    '''Fully Conncted ICNN with input-quadratic skip connections'''
    def __init__(
        self, in_dim_a, in_dim_b, 
        hidden_layer_sizes=[32, 32, 32],
        rank=1, activation='celu', 
        strong_convexity=1e-6,device = 'cuda:0'
    ):
        super(DynamicDenseICNN, self).__init__()
        self.device = device
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.rank = rank
        self.strong_convexity = strong_convexity
        
        # a output shape
        
        self.quadratic_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim_a, out_features, bias=True),
            )
            for out_features in self.hidden_layer_sizes
        ])
        self.de_dim_b = 32
        self.sizes = zip(self.hidden_layer_sizes[:-1], self.hidden_layer_sizes[1:])

        # 我们动态来用凸层
        self.convex_layers =  nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
            )
            for (in_features, out_features) in self.sizes
        ])
        self.de_dim = nn.Linear(in_dim_b,self.de_dim_b )
        self.b_path =  nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.de_dim_b, in_features, bias=True),
            )
            for (in_features, out_features) in self.sizes
        ])
        if self.activation == 'celu':
            self.act = torch.nn.CELU()
        elif self.activation == 'softplus':
            self.act = torch.nn.Softplus()

        elif self.activation == "leakyrelu":
            self.act = torch.nn.LeakyReLU()
        else:
            raise Exception('Activation is not specified or unknown.')
        self.final_layer = nn.Linear(hidden_layer_sizes[-1], 1, bias=False)

    def forward(self, input_a, input_b):
        generator = self.de_dim(input_b)
        # after convex_layers are calculated
        output = self.quadratic_layers[0](input_a)
        for quadratic_layer, convex_layer, b_path_layer in zip(self.quadratic_layers[1:], self.convex_layers, self.b_path):
            output = convex_layer(output ) + quadratic_layer(input_a) + b_path_layer(generator) 
            #output = convex_layer(output + b_path_layer(generator) ) + quadratic_layer(input_a)
            output = self.act(output)
        return self.final_layer(output)
    
    def pred(self, in_a, in_b):
        return self.push(input_a=in_a, input_b = in_b)
    def push(self, input_a, input_b):
        output = autograd.grad(
            outputs=self.forward(input_a,input_b), inputs=input_a,
            create_graph=True, retain_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((input_a.size()[0], 1)).to(device=self.device).float()
        )[0]
        return output    
    
    def convexify(self):
        for layer in self.convex_layers:
            for sublayer in layer:
                if (isinstance(sublayer, nn.Linear)):
                    sublayer.weight.data.clamp_(0)

        self.final_layer.weight.data.clamp_(0)
class DICNN(nn.Module):
    '''Fully Conncted ICNN with input-quadratic skip connections'''
    def __init__(
        self, in_dim, 
        hidden_layer_sizes=[32, 32, 32],
        rank=1, activation='celu', 
        strong_convexity=1e-6,device = 'cuda:0'
    ):
        super(DICNN, self).__init__()
        self.device = device
        self.strong_convexity = strong_convexity
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.rank = rank
        self.icnn = OriDenseICNN(hidden_layer_sizes[0], hidden_layer_sizes[1:], rank, activation, strong_convexity, device)
        self.l = nn.Linear(in_dim, hidden_layer_sizes[0], bias=True)
    def forward(self, input):
        o = self.l(input)
        o = self.icnn(o)
        return o
    def push(self, input):
        output = autograd.grad(
            outputs=self.forward(input), inputs=input,
            create_graph=True, retain_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((input.size()[0], 1)).to(device=self.device).float()
        )[0]
        return output    
    
    def convexify(self):
        self.icnn.convexify()

class IOCNN(nn.Module):
    '''Fully Conncted ICNN with input-quadratic skip connections'''
    def __init__(
        self, in_dim, 
        hidden_layer_sizes=[32, 32, 32],
         activation='elu',
         convexify = 'exp', 
        device = 'cuda:0'
    ):
        super(IOCNN, self).__init__()
        self.device = device
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.strong_convexity = 1e-6
        self.quadratic_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_features, bias=True),
            )
            for out_features in self.hidden_layer_sizes
        ])
        self.eps = 5
        sizes = zip(self.hidden_layer_sizes[:-1], self.hidden_layer_sizes[1:])
        self.convex_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
            )
            for (in_features, out_features) in sizes
        ])
        self.half_input =int( in_dim /2); 
        if self.activatFion == 'celu':
            self.act = torch.nn.CELU()
        elif self.activation == 'softplus':
            self.act = torch.nn.Softplus()
        elif self.activation == 'elu':
            self.act = torch.nn.ELU()
        elif self.activation == "leakyrelu":
            self.act = torch.nn.LeakyReLU()
        else:
            raise Exception('Activation is not specified or unknown.')
        self.final_layer = nn.Linear(hidden_layer_sizes[-1], 1, bias=False)
        if convexify == 'exp':
            self.convexify = self.convexify1
        else:
            self.convexify = self.convexify2
    def forward(self, input_a, input_b):
        input = torch.cat((input_a, input_b), dim = -1)
        output = self.quadratic_layers[0](input)
        for quadratic_layer, convex_layer in zip(self.quadratic_layers[1:], self.convex_layers):
            output = convex_layer(output) + quadratic_layer(input)
            output = self.act(output)
        return self.final_layer(output) + .5 * self.strong_convexity * (input ** 2).sum(dim=1).reshape(-1, 1)
    
    def push(self, input_a, input_b):
        output = autograd.grad(
            outputs=self.forward(input_a,input_b), inputs=input_a,
            create_graph=True, retain_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((input_a.size()[0], 1)).to(device=self.device).float()
        )[0]
        return output    
    def pred(self, input_a,input_b):
        f_pred = self.push(input_a,input_b)
        return f_pred - f_pred.mean(dim=1, keepdim = True)
    def convexify1(self):
        for layer in self.convex_layers:
            for sublayer in layer:
                if (isinstance(sublayer, nn.Linear)):
                    w = sublayer.weight.data
                    w = (w<0) * torch.exp(w - self.eps) + (w>=0) * w
                    sublayer.weight.data = w
        self.final_layer.weight.data.clamp_(0)
    def convexify2(self):
        for layer in self.convex_layers:
            for sublayer in layer:
                if (isinstance(sublayer, nn.Linear)):
                    sublayer.weight.data.clamp_(0)

        self.final_layer.weight.data.clamp_(0)
class IOCNN(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_num = 3, hidden_dim = 1024 ,eps = 5,device = torch.device("cuda"), act = nn.ReLU()):
        super().__init__()
        self.eps = eps
        self.device = device
        self.mlp = PotentialMLP(dim_in, dim_out, hidden_num=hidden_num,hidden_dim= hidden_dim, act = act)
    def forward(self, a, b):
        return self.mlp(a, b)
    def push(self, input_a, input_b):
        output = autograd.grad(
            outputs=self.forward(input_a,input_b), inputs=input_a,
            create_graph=True, retain_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((input_a.size()[0], 1)).to(device=self.device).float()
        )[0]
        return output    
    def pred(self, input_a,input_b):
        f = self.push(input_a,input_b)
        
        return f - f.mean(dim=1, keepdim = True) 
    def convexify(self):
        for layer in self.mlp.model_base[1:]:
            for sublayer in layer:
                if isinstance(sublayer, nn.Linear):
                    w = sublayer.weight.data # is this a ref?
                    w = (w<0) * torch.exp(w - self.eps) + (w>=0) * w
                    sublayer.weight.data = w
                    
        w = self.mlp.lin.weight.data
        w = (w<0) * torch.exp(w - self.eps) + (w>=0) * w
        self.mlp.lin.weight.data= w
        return
import time

class ot_solver():
    def __init__(self, reg, K, C, logger, device):
        self.reg = self.epsilon = reg
        self.K = K
        self.C = C
        self.logger = logger
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
            self.logger.warn("numerical error nan")
        if torch.any(torch.isinf(g_sink)) or torch.any(torch.isinf(f_sink)):
            self.logger.warn("numerical error nan")
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
    def grad_loss(self, a,b,f):
        g_sink, f_sink = self.update(a, b, f)
        return self.grad_value(a,b,f_sink,g_sink)
    def potential_loss(self, a, b, f_pred, g_pred = None):
        if(g_pred is None):
            g_sink, f_sink = self.update(a, b, f_pred)
        else:
            g_sink = g_pred; f_sink = f_pred
        if torch.any(torch.isnan(g_sink)) or torch.any(torch.isnan(f_sink)):
            self.logger.warn("numerical error nan")
        if torch.any(torch.isinf(g_sink)) or torch.any(torch.isinf(f_sink)):
            self.logger.warn("numerical error nan")
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
    def initvalue(self, a,b,init = None):
        if init is None : 
            f = torch.zeros((a.shape[0],self.K.shape[0]),dtype=torch.float64, device= self.device)
        else:
            f = init
        g,f = self.update(a,b,f)
        b0 =  torch.exp(g/self.reg)*(torch.exp(f/self.reg)@(self.K))
        norm2 = torch.norm((b0 - b),dim = 1,keepdim=True)
        return torch.sum(norm2).detach().item()
    def otvalue(self,f,g):
        P = self.P(f,g)
        return torch.sum(P*self.C,dim = (1,2)) + self.epsilon*torch.sum(torch.xlogy(P,P/torch.e),dim = (1,2))
class uot_solver():
    def __init__(self, reg, K,C,device, logger, rho = 1):
        self.reg = self.epsilon  = reg
        self.K = K
        self.device = device
        self.C = C
        self.rho = rho
        self.logger = logger
        self.coeff = self.rho*self.epsilon/(self.epsilon+self.rho)
        logger.info("pho : {}".format(self.rho))
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
            self.logger.info("numerical error nan")
            return None,None,None
        if torch.any(torch.isinf(g_sink)) or torch.any(torch.isinf(f_sink)):
            self.logger.info("numerical error nan")
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
        return torch.sum(P*self.C,dim = (1,2)) + self.epsilon*torch.sum(torch.xlogy(P,P/torch.e),dim = (1,2))
    
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
        im_grad_1 = ((source_count) / (source_count * source_count) * f ) # size of [#batch * #cls]
        im_grad_2 = (source_unnorm * f).sum(dim = 1, keepdim = True)/ (source_count * source_count)
        im_grad = (im_grad_1 - im_grad_2).detach()
        # Define loss = <im_grad, predicted density>. The gradient of loss w.r.t prediced density is im_grad.
        loss = torch.sum(a * im_grad)
        return loss
    def potential_loss(self, a, b, f_pred):
        return self.solver.potential_loss(a,b,f_pred)
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
        return loss,f.detach()
    def potential_loss(self,a,b,f,g = None):
        return self.solver.potential_loss(a,b,f,g_pred= g)
    def grad_loss(self, a,b,f):
        return self.solver.grad_loss(a,b,f,g = None)
    def regress_loss(self,f,real_f):
        return torch.norm(f-real_f,dim = 1,keepdim=True).mean()
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
        return loss
    def potential_loss(self,a,b,f):
        return self.solver.potential_loss(a,b,f)