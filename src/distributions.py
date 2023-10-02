import torch
import numpy as np
import random
from scipy.linalg import sqrtm
from sklearn import datasets

class Sampler:
    def __init__(
        self, device='cuda',
    ):
        self.device = device
    
    def sample(self, size=5):
        pass
    
class LoaderSampler(Sampler):
    def __init__(self, loader, device='cuda'):
        super(LoaderSampler, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)
        
    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            batch, _ = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch) < size:
            return self.sample(size)
            
        return batch[:size].to(self.device)
    

class TensorSampler(Sampler):
    def __init__(self, tensor, device='cuda'):
        super(TensorSampler, self).__init__(device)
        self.tensor = torch.clone(tensor).to(device)
        
    def sample(self, size=5):
        assert size <= self.tensor.shape[0]
        
        ind = torch.tensor(np.random.choice(np.arange(self.tensor.shape[0]), size=size, replace=False), device=self.device)
        return torch.clone(self.tensor[ind]).detach().to(self.device)


class SwissRollSampler(Sampler):
    def __init__(
        self, dim=2, device='cuda'
    ):
        super(SwissRollSampler, self).__init__(device=device)
        assert dim == 2
        self.dim = 2
        
    def sample(self, batch_size=10):
        batch = datasets.make_swiss_roll(
            n_samples=batch_size,
            noise=0.8
        )[0].astype('float32')[:, [0, 2]] / 7.5
        return torch.tensor(batch, device=self.device)
    
    
class StandardNormalSampler(Sampler):
    def __init__(self, dim=1, device='cuda'):
        super(StandardNormalSampler, self).__init__(device=device)
        self.dim = dim
        
    def sample(self, batch_size=10):
        return torch.randn(batch_size, self.dim, device=self.device)
    
    
class SwissRollSampler(Sampler):
    def __init__(
        self, dim=2, device='cuda'
    ):
        super(SwissRollSampler, self).__init__(device=device)
        assert dim == 2
        self.dim = 2
        
    def sample(self, batch_size=10):
        batch = datasets.make_swiss_roll(
            n_samples=batch_size,
            noise=0.8
        )[0].astype('float32')[:, [0, 2]] / 7.5
        return torch.tensor(batch, device=self.device)
    
    
class Mix8GaussiansSampler(Sampler):
    def __init__(self, with_central=False, std=1, r=12, dim=2, device='cuda'):
        super(Mix8GaussiansSampler, self).__init__(device=device)
        assert dim == 2
        self.dim = 2
        self.std, self.r = std, r
        
        self.with_central = with_central
        centers = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        if self.with_central:
            centers.append((0, 0))
        self.centers = torch.tensor(centers, device=self.device, dtype=torch.float32)
        
    def sample(self, batch_size=10):
        with torch.no_grad():
            batch = torch.randn(batch_size, self.dim, device=self.device)
            indices = random.choices(range(len(self.centers)), k=batch_size)
            batch *= self.std
            batch += self.r * self.centers[indices, :]
        return batch

class Transformer(object):
    def __init__(self, device='cuda'):
        self.device = device
        

class StandardNormalScaler(Transformer):
    def __init__(self, base_sampler, batch_size=1000, device='cuda'):
        super(StandardNormalScaler, self).__init__(device=device)
        self.base_sampler = base_sampler
        batch = self.base_sampler.sample(batch_size).cpu().detach().numpy()
        
        mean, cov = np.mean(batch, axis=0), np.cov(batch.T)
        
        self.mean = torch.tensor(
            mean, device=self.device, dtype=torch.float32
        )
        
        multiplier = sqrtm(cov)
        self.multiplier = torch.tensor(
            multiplier, device=self.device, dtype=torch.float32
        )
        self.inv_multiplier = torch.tensor(
            np.linalg.inv(multiplier),
            device=self.device, dtype=torch.float32
        )
        torch.cuda.empty_cache()
        
    def sample(self, batch_size=10):
        with torch.no_grad():
            batch = torch.tensor(self.base_sampler.sample(batch_size), device=self.device)
            batch -= self.mean
            batch @= self.inv_multiplier
        return batch
    
class LinearTransformer(Transformer):
    def __init__(
        self, base_sampler, weight, bias=None,
        device='cuda'
    ):
        super(LinearTransformer, self).__init__(device=device)
        self.base_sampler = base_sampler
        
        self.weight = torch.tensor(weight, device=device, dtype=torch.float32)
        if bias is not None:
            self.bias = torch.tensor(bias, device=device, dtype=torch.float32)
        else:
            self.bias = torch.zeros(self.weight.size(0), device=device, dtype=torch.float32)
        
    def sample(self, size=4):        
        batch = torch.tensor(
            self.base_sampler.sample(size),
            device=self.device
        )
        with torch.no_grad():
            batch = batch @ self.weight.T
            if self.bias is not None:
                batch += self.bias
        return batch

class StandartNormalSampler(Sampler):
    def __init__(
        self, dim=1, device='cuda',
        dtype=torch.float, requires_grad=False
    ):
        super(StandartNormalSampler, self).__init__(
            device=device
        )
        self.requires_grad = requires_grad
        self.dtype = dtype
        self.dim = dim
        
    def sample(self, batch_size=10):
        return torch.randn(
            batch_size, self.dim, dtype=self.dtype,
            device=self.device, requires_grad=self.requires_grad
        )
