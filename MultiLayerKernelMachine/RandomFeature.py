import torch

def sample_1d(pdf, gamma, device):
    if pdf == 'G':
        w = torch.randn(1, device=device) * gamma
        return w
    elif pdf == 'L':
        w = torch.distributions.laplace.Laplace(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device)).sample() * gamma
        return w
    elif pdf == 'C':
        w = torch.distributions.cauchy.Cauchy(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device)).sample() * gamma
        return w
    
def sample(pdf, gamma, d, device):
    return torch.tensor([sample_1d(pdf, gamma, device) for _ in range(d)], device=device)

class RandomFourierFeature:
    """Random Fourier Feature
    Parameters
    ----------
    d : int
        Input space dimension
    D : int
        Feature space dimension
    W : shape (D,d)
    b : shape (D)
    kernel : char
        Kernel to use; 'G', 'L', or 'C'
    gamma : float
        Kernel scale
    """
    
    def __init__(self, d, D, W=None, b=None, kernel='G', gamma=1, device='cpu'):
        self.d = d
        self.D = D
        self.gamma = gamma
        self.device = device

        kernel = kernel.upper()
        if kernel not in ['G', 'L', 'C']:
            raise Exception('Invalid Kernel')
        self.kernel = kernel
        self.create()
    
    
    def create(self):
        """Create a d->D fourier random feature"""
        self.b = torch.rand(self.D, device=self.device) * 2 * torch.pi
        self.W = sample(self.kernel, self.gamma, self.d * self.D, self.device).reshape(self.D, self.d)

    def transform(self, x):
        """Transform a vector using this feature
        Parameters
        ----------
        x : (shape=(n,d))
            to transform; 
            
        Returns
        -------
        x : (shape=(n,D))
            Feature space transformation of x
        """
       

        result = torch.sqrt(torch.tensor([2.0 / self.D], device=x.device)) * torch.cos(
            self.W @ x.T + (self.b.reshape(-1, 1) @ torch.ones((1,len(x)), device=x.device))
        )
        return result.T