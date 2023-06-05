from geometric_solvers import GeomtricFixedGridODESolver
from torchdiffeq._impl.misc import Perturb
import torch


class SymplecticEuler(GeomtricFixedGridODESolver):
    order = 1

    def _step_func(self, func, t0, dt, t1, y0):
        x0, z0 = torch.split(y0, y0.shape[1] // 2, dim=1)
        f0x = torch.split(func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE),y0.shape[1] // 2, dim=1)[0]
        x1 = x0 + dt*f0x
        f0z = torch.split(func(t0, torch.cat([x1,z0],dim=1), perturb=Perturb.NEXT if self.perturb else Perturb.NONE), y0.shape[1] // 2, dim=1)[1]
        z1 = z0 + dt*f0z
        y1 = torch.cat([x1,z1],dim=1)
        return y1

class Leapfrog(GeomtricFixedGridODESolver):
    order = 2

    def _step_func(self, func, t0, dt, t1, y0):
        x0, z0 = torch.split(y0, y0.shape[1] // 2, dim=1)
        f0x_1 = torch.split(func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE),y0.shape[1] // 2, dim=1)[0]
        x_half = x0 + dt/2.*f0x_1

        f0z = torch.split(func(t0, torch.cat([x_half, z0], dim=1), perturb=Perturb.NEXT if self.perturb else Perturb.NONE),y0.shape[1] // 2, dim=1)[1]
        z1 = z0 + dt * f0z

        f0x_2 = torch.split(func(t0, torch.cat([x_half, z0], dim=1), perturb=Perturb.NEXT if self.perturb else Perturb.NONE), y0.shape[1] // 2, dim=1)[0]
        x1 = x_half + dt/2.*f0x_2
        y1 = torch.cat([x1,z1],dim=1)
        return y1


def SymplecticEuler_step_func(func, t0, dt, t1, y0, perturb=False):
    x0, z0 = torch.split(y0, y0.shape[1] // 2, dim=1)
    f0x = torch.split(func(t0, y0, perturb=Perturb.NEXT if perturb else Perturb.NONE),y0.shape[1] // 2, dim=1)[0]
    x1 = x0 + dt*f0x
    y0 = torch.cat([x1,z0],dim=1)
    f0z = torch.split(func(t0, y0, perturb=Perturb.NEXT if perturb else Perturb.NONE), y0.shape[1] // 2, dim=1)[1]
    z1 = z0 + dt*f0z
    y1 = torch.cat([x1,z1],dim=1)
    return y1


def Leapfrog_step_func(func, t0, dt, t1, y0, perturb=False):
    x0, z0 = torch.split(y0, y0.shape[1] // 2, dim=1)
    f0x_1 = torch.split(func(t0, y0, perturb=Perturb.NEXT if perturb else Perturb.NONE), y0.shape[1] // 2, dim=1)[0]
    x_half = x0 + dt / 2. * f0x_1

    f0z = torch.split(func(t0, torch.cat([x_half, z0], dim=1), perturb=Perturb.NEXT if perturb else Perturb.NONE),y0.shape[1] // 2, dim=1)[1]
    z1 = z0 + dt * f0z

    f0x_2 = torch.split(func(t0, torch.cat([x_half, z0], dim=1), perturb=Perturb.NEXT if perturb else Perturb.NONE),y0.shape[1] // 2, dim=1)[0]
    x1 = x_half + dt / 2. * f0x_2
    y1 = torch.cat([x1, z1], dim=1)
    return y1