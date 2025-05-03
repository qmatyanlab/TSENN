import numpy as np

def cart2msph(cart):
    x=0
    y=1
    z=2
    msph = {
        '*s_0^0': cart[x][x] + cart[y][y] + cart[z][z] ,
        '*s_2^-2': (3/2)**(1/2)*(0.5*(cart[x][x] - cart[y][y]) - 1j*cart[x][y]) ,
        '*s_2^-1': (3/2)**(1/2)*(cart[x][z] - 1j*cart[y][z]) ,
        '*s_2^0': -0.5*(cart[x][x] + cart[y][y]) + cart[z][z] ,
        '*s_2^1': (3/2)**(1/2)*(-cart[x][z] - 1j*cart[y][z]),
        '*s_2^2': (3/2)**(1/2)*(0.5*(cart[x][x] - cart[y][y]) + 1j*cart[x][y])
    }

    return msph

def msph2sph(msph):
    sph = {
        's_0^0' : ((1/(4*np.pi))**(1/2))*msph['*s_0^0'],
        's_2^-2' : ((5/(4*np.pi))**(1/2))*msph['*s_2^-2'],
        's_2^-1' : ((5/(4*np.pi))**(1/2))*msph['*s_2^-1'],
        's_2^0' : ((5/(4*np.pi))**(1/2))*msph['*s_2^0'],
        's_2^1' : ((5/(4*np.pi))**(1/2))*msph['*s_2^1'],
        's_2^2' : ((5/(4*np.pi))**(1/2))*msph['*s_2^2']
    }
    return sph

def cart2sph(cart):
    msph = cart2msph(cart)
    sph = msph2sph(msph)
    
    return sph

def cart2sphvec(cart):
    sph = cart2sph(cart)
    sphvec = np.array([
        sph['s_0^0'],
        sph['s_2^-2'],
        sph['s_2^-1'],
        sph['s_2^0'],
        sph['s_2^1'],
        sph['s_2^2']
        ])
    return sphvec

def sphvec2sph(sphvec):
    sph = {
        's_0^0' : sphvec[0],
        's_2^-2' : sphvec[1],
        's_2^-1' : sphvec[2],
        's_2^0' : sphvec[3],
        's_2^1' : sphvec[4],
        's_2^2' : sphvec[5]
    }
    return sph


def sph2realsph(sph):
    realsph = {}
    realsph['s_0_0'] = sph['s_0^0']

    realsph['s_2_-2'] = (1j/(2**(1/2)))*(sph['s_2^-2']-sph['s_2^2'])
    realsph['s_2_-1'] = (1j/(2**(1/2)))*(sph['s_2^-1']+sph['s_2^1'])
    realsph['s_2_0'] = sph['s_2^0']
    realsph['s_2_1'] = (1/(2**(1/2)))*(sph['s_2^-1']-sph['s_2^1'])
    realsph['s_2_2'] = (1/(2**(1/2)))*(sph['s_2^-2']+sph['s_2^2'])

    return realsph

def cart2realsph(cart):
    msph = cart2msph(cart)
    sph = msph2sph(msph)
    realsph = sph2realsph(sph)

    return realsph

def cart2realsphvec(cart):
    realsph = cart2realsph(cart)
    realsphvec = np.array([
        realsph['s_0_0'],
        realsph['s_2_-2'],
        realsph['s_2_-1'],
        realsph['s_2_0'],
        realsph['s_2_1'],
        realsph['s_2_2']
        ])

    return realsphvec

def msph2cart(msph):
    x=0
    y=1
    z=2
    cart = np.array([[0.0,0,0],[0,0,0],[0,0,0]])

    cart[x][x] = (1/2)*((2/3)*(msph['*s_0^0']-msph['*s_2^0'])+(2/3)**(1/2)*(msph['*s_2^2']+msph['*s_2^-2']))
    cart[y][y] = (1/2)*((2/3)*(msph['*s_0^0']-msph['*s_2^0'])-(2/3)**(1/2)*(msph['*s_2^2']+msph['*s_2^-2']))
    cart[z][z] = (1/3)*(msph['*s_0^0']+2*msph['*s_2^0'])
    cart[x][z] = cart[z][x] = (1/2)*((2/3)**(1/2))*(msph['*s_2^-1']-msph['*s_2^1'])
    cart[y][z] = cart[z][y] = 1j*(1/2)*(2/3)**(1/2)*(msph['*s_2^-1']+msph['*s_2^1'])
    cart[x][y] = cart[y][x] = 1j*(1/2)*(2/3)**(1/2)*(msph['*s_2^-2']-msph['*s_2^2'])

    return cart

def sph2msph(sph):
    msph = {
        '*s_0^0' : ((1/(4*np.pi))**(-1/2))*sph['s_0^0'],
        '*s_2^-2' : ((5/(4*np.pi))**(-1/2))*sph['s_2^-2'],
        '*s_2^-1' : ((5/(4*np.pi))**(-1/2))*sph['s_2^-1'],
        '*s_2^0' : ((5/(4*np.pi))**(-1/2))*sph['s_2^0'],
        '*s_2^1' : ((5/(4*np.pi))**(-1/2))*sph['s_2^1'],
        '*s_2^2' : ((5/(4*np.pi))**(-1/2))*sph['s_2^2']
    }

    return msph


def sphvec2cart(sphvec):
    sph = sphvec2sph(sphvec)
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

def realsph2cart(realsph):
    sph = realsph2sph(realsph)
    msph = sph2msph(sph)

    return msph2cart(msph)

def realsphvec2cart(realsphvec):
    realsph = {}
    realsph['s_0_0'] = realsphvec[0]

    realsph['s_2_-2'] = realsphvec[1]
    realsph['s_2_-1'] = realsphvec[2]
    realsph['s_2_0'] = realsphvec[3]
    realsph['s_2_1'] = realsphvec[4]
    realsph['s_2_2'] = realsphvec[5]

    cart = realsph2cart(realsph)

    return cart