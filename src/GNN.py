import torch
import torch.nn.functional as F
from base_classes import BaseGNN
from model_configurations import set_block, set_function


# Define the GNN model.
class GNN(BaseGNN):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(GNN, self).__init__(opt, dataset, device)
    self.f = set_function(opt)
    block = set_block(opt)
    time_tensor = torch.tensor([0, self.T]).to(device)
    self.odeblock = block(self.f, self.regularization_fns, opt, dataset.data, device, t=time_tensor).to(device)
    self.odeblock.odefunc.GNN_postXN = self.GNN_postXN
    self.odeblock.odefunc.GNN_m2 = self.m2

    if opt['trusted_mask']:
        self.trusted_mask = dataset.data.train_mask.to(device)
    else:
        self.trusted_mask = None

  def encoder(self, x, pos_encoding=None):
    # Encode each node based on its feature.
    if self.opt['use_labels']:
      y = x[:, -self.num_classes:]
      x = x[:, :-self.num_classes]

    x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    if self.opt['beltrami']:
      x = self.mx(x)
      p = self.mp(pos_encoding)
      x = torch.cat([x, p], dim=1)
    else:
      x = self.m1(x)

    if self.opt['use_mlp']==True:
      x = F.dropout(x, self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)

    if self.opt['use_labels']:
      x = torch.cat([x, y], dim=-1)

    if self.opt['batch_norm']==True:
      x = self.bn_in(x)

    if self.opt['augment']==True:
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)

    return x

  # def set_attributes(self, x):
  #   self.odeblock.odefunc.W = self.odeblock.odefunc.set_W()
  #   self.odeblock.odefunc.Beta = self.odeblock.odefunc.set_Beta()

  def forward_XN(self, x, pos_encoding=None):
    ###forward XN
    x = self.encoder(x, pos_encoding)
    self.odeblock.set_x0(x)
    if self.opt['function']=='gread':
      if self.opt['beta_diag'] == True:
        self.odeblock.odefunc.Beta = self.odeblock.odefunc.set_Beta()

    if self.trusted_mask is not None:
      if self.opt['nox0']==True:
        self.odeblock.set_x0(x*0)
      xp = x * self.trusted_mask[:, None]
      ave = xp.sum(dim=0) / self.trusted_mask.sum(dim=0)
      xp -= ave[None, :]
      xp *= self.trusted_mask[:, None]
      x += xp

    if self.training and self.odeblock.nreg > 0:
      z, self.reg_states = self.odeblock(x)
    else:
      z = self.odeblock(x)
    return z

  def GNN_postXN(self, z):
    if self.opt['augment']==True:
      z = torch.split(z, z.shape[1] // 2, dim=1)[0]
    # Activation.
    if self.opt['XN_activation']==True:
      z = F.relu(z)
    # fc from bottleneck
    if self.opt['fc_out']==True:
      z = self.fc(z)
      z = F.relu(z)
    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)
    return z

  def forward(self, x, pos_encoding=None):
    z = self.forward_XN(x,pos_encoding)
    z = self.GNN_postXN(z)
    # Decode each node embedding to get node label.
    z = self.m2(z)
    return z