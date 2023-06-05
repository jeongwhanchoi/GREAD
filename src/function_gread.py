import torch
from torch import nn
import torch_sparse
from torch.nn.init import uniform, xavier_uniform_
from base_classes import ODEFunc
from utils import MaxNFEException


"""
Define the ODE function.
Input:
 - t: A tensor with shape [], meaning the current time.
 - x: A tensor with shape [#nodes, dims], meaning the value of x at t.
Output:
 - dx/dt: A tensor with shape [#nodes, dims], meaning the derivative of x at t.
"""
class ODEFuncGread(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(ODEFuncGread, self).__init__(opt, data, device)

    self.in_features = in_features
    self.out_features = out_features

    self.reaction_tanh = False
    if opt['beta_diag'] == True:
      self.b_W = nn.Parameter(torch.Tensor(in_features))
      self.reset_parameters()
    self.epoch = 0
  
  def reset_parameters(self):
    if self.opt['beta_diag'] == True:
      uniform(self.b_W, a=-1, b=1)
  
  def set_Beta(self, T=None):
    Beta = torch.diag(self.b_W)
    return Beta

  def sparse_multiply(self, x):
    """
    - `attention` is equivalent to "Soft Adjacency Matrix (SA)".
    - If `block` is `constant`, we use "Original Adjacency Matrix (OA)"
    """
    if self.opt['block'] in ['attention']:  # adj is a multihead attention
      mean_attention = self.attention_weights.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    else:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
    return ax

  def forward(self, t, x):  # the t param is needed by the ODE solver.
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1
    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
      beta = torch.sigmoid(self.beta_train)
    else:
      alpha = self.alpha_train
      beta = self.beta_train

    """
    - `x` is equivalent $H$ in our paper.
    - `diffusion` is the diffusion term.
    """
    ax = self.sparse_multiply(x)
    diffusion = (ax - x)

    """
    - `reaction` is the reaction term.
    - We consider four `reaction_term` options
     - When `reaction_term` is bspm: GREAD-BS
     - When `reaction_term` is fisher: GREAD-F
     - When `reaction_term` is allen-cahn: GREAD-AC
     - When `reaction_term` is zeldovich: GREAD-Z
    - The `tanh` on reaction variable is optional, but we don't use in our experiments.
    """
    if self.opt['reaction_term'] == 'bspm':
      reaction = -self.sparse_multiply(diffusion) # A(AX-X)
    elif self.opt['reaction_term'] == 'fisher':
      reaction = -(x-1)*x
      if self.reaction_tanh == True:
        reaction = torch.tanh(reaction)
    elif self.opt['reaction_term'] == 'allen-cahn':
      reaction = -(x**2-1)*x
      if self.reaction_tanh == True:
        reaction = torch.tanh(reaction)
    elif self.opt['reaction_term'] == 'zeldovich':
      reaction = -(x**2-x)*x
      if self.reaction_tanh == True:
        reaction = torch.tanh(reaction)
    elif self.opt['reaction_term'] =='st':
      reaction = self.x0
    elif self.opt['reaction_term'] == 'fb':
      ax = -self.sparse_multiply(x)
      reaction = x - ax # L = I - A
      if self.reaction_tanh == True:
        reaction = torch.tanh(reaction)
    elif self.opt['reaction_term'] == 'fb3':
      ax = -self.sparse_multiply(x)
      reaction = x - ax# L = I - A
      if self.reaction_tanh == True:
        reaction = torch.tanh(reaction)
    else:
      reaction = 0.0
    
    """
    - `f` is in the reaction-diffusion form
     - $\mathbf{f}(\mathbf{H}(t)) := \frac{d \mathbf{H}(t)}{dt} = -\alpha\mathbf{L}\mathbf{H}(t) + \beta\mathbf{r}(\mathbf{H}(t), \mathbf{A})$
    - `beta_diag` is equivalent to $\beta$ with VC dimension
     - `self.Beta` is diagonal matrix intialized with gaussian distribution
     - Due to the diagonal matrix, it is same to the result of `beta*reaction` when `beta` is initialized with gaussian distribution.
    """
    if self.opt['beta_diag'] == False:
      if self.opt['reaction_term'] =='fb':
        f = alpha*diffusion + beta*reaction
      elif self.opt['reaction_term'] =='fb3':
        f = alpha*diffusion + beta*(reaction + x)
      else:
        f = alpha*diffusion + beta*reaction
    elif self.opt['beta_diag'] == True:
      f = alpha*diffusion + reaction@self.Beta
    
    """
    - We do not use the `add_source` on GREAD
    """
    if self.opt['add_source']:
      f = f + self.source_train * self.x0
    return f
