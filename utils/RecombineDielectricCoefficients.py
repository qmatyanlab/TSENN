import torch
import warnings

def realsphvec2cart(realsphvec):
    realsph = {}
    realsph['s_0_0'] = realsphvec[:, 0]

    realsph['s_2_-2'] = realsphvec[:, 1]
    realsph['s_2_-1'] = realsphvec[:, 2]
    realsph['s_2_0'] = realsphvec[:, 3]
    realsph['s_2_1'] = realsphvec[:, 4]
    realsph['s_2_2'] = realsphvec[:, 5]

    cart = realsph2cart(realsph)

    return cart

def realsph2cart(realsph):
    sph = realsph2sph(realsph)
    msph = sph2msph(sph)

    return msph2cart(msph)

def realsph2sph(realsph):
    sph = {
        's_0^0': realsph['s_0_0'],

        's_2^-2': (2**(-1/2))*(realsph['s_2_2']-1j*realsph['s_2_-2']),
        's_2^-1': (2**(-1/2))*(realsph['s_2_1']-1j*realsph['s_2_-1']),
        's_2^0': realsph['s_2_0'] ,
        's_2^1': -(1/(2**(1/2)))*(realsph['s_2_1']+1j*realsph['s_2_-1']),
        's_2^2': (1/(2**(1/2)))*(realsph['s_2_2']+1j*realsph['s_2_-2']),
    }

    return sph



def sph2msph(sph):
    msph = {
        '*s_0^0' : ((1/(4*torch.pi))**(-1/2))*sph['s_0^0'],
        '*s_2^-2' : ((5/(4*torch.pi))**(-1/2))*sph['s_2^-2'],
        '*s_2^-1' : ((5/(4*torch.pi))**(-1/2))*sph['s_2^-1'],
        '*s_2^0' : ((5/(4*torch.pi))**(-1/2))*sph['s_2^0'],
        '*s_2^1' : ((5/(4*torch.pi))**(-1/2))*sph['s_2^1'],
        '*s_2^2' : ((5/(4*torch.pi))**(-1/2))*sph['s_2^2']
    }

    return msph

def msph2cart(msph):
    x=0
    y=1
    z=2
    cart = torch.zeros((msph['*s_0^0'].shape[0], 3, 3), dtype=torch.complex64, device=msph['*s_0^0'].device)

    cart[:, x, x] = (1/2)*((2/3)*(msph['*s_0^0']-msph['*s_2^0'])+(2/3)**(1/2)*(msph['*s_2^2']+msph['*s_2^-2']))
    cart[:, y, y] = (1/2)*((2/3)*(msph['*s_0^0']-msph['*s_2^0'])-(2/3)**(1/2)*(msph['*s_2^2']+msph['*s_2^-2']))
    cart[:, z, z] = (1/3)*(msph['*s_0^0']+2*msph['*s_2^0'])
    cart[:, x, z] = cart[:, z, x] = (1/2)*((2/3)**(1/2))*(msph['*s_2^-1']-msph['*s_2^1'])
    cart[:, y, z] = cart[:, z, y] = 1j*(1/2)*(2/3)**(1/2)*(msph['*s_2^-1']+msph['*s_2^1'])
    cart[:, x, y] = cart[:, y, x] = 1j*(1/2)*(2/3)**(1/2)*(msph['*s_2^-2']-msph['*s_2^2'])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cart = cart.float()

    return cart