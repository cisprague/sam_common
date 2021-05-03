# 6 degree of freedom Fossen AUV model
# Christopher Iliffe Sprague

import jax.numpy as np
from jax import jit, jacfwd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from dynamics import Dynamics
from collections import OrderedDict

class Fossen(Dynamics):

    # state and control dimensionality
    state_dim = 12
    control_dim = 4

    # state bounds and scaling
    state_lb = np.array([
        *[-2000]*3, # position
        *[-np.pi]*3,      # quaternions
        *[-10]*3,     # velocity
        *[-10]*3      # ang. velocity
    ])
    state_ub = -state_lb

    # control bounds
    control_lb = np.array([*[-200]*2, *[-0.15]*2])
    control_ub = np.array([*[1500]*2, *[0.15]*2])

    # default parameters
    params = OrderedDict(
        Nrr=150., Izz=10., Kt1=0.1, zb=0., Mqq=100., 
        ycp=0., xb=0., zcp=0., Yvv=100., yg=0., Ixx=10., 
        Kt0=0.1, Xuu=1., xg=0., Zww=100., W=15.4*9.81, 
        m=15.4, B=15.4*9.81, zg=0., Kpp=100., Qt1=-0.001, 
        Qt0=0.001, Iyy=10., yb=0., xcp=0.1
    )

    def __init__(self, **kwargs):

        # become dynamics model
        Dynamics.__init__(self, **kwargs)

    @staticmethod
    # @jit
    def state_dynamics(state, control, *params):

        # sanity
        assert len(state.shape) == len(control.shape) == 1
        assert state.shape[-1] == Fossen.state_dim
        assert control.shape[-1] == Fossen.control_dim

        # constant parameters
        Nrr, Izz, Kt1, zb, Mqq, ycp, xb, zcp, Yvv, yg, Ixx, Kt0, Xuu, xg, Zww, W, m, B, zg, Kpp, Qt1, Qt0, Iyy, yb, xcp = params

        # extract state and control
        # phi, theta, psi = roll, pitch, yaw
        x, y, z, phi, theta, psi, u, v, w, p, q, r = state
        rpm0, rpm1, de, dr = control
        # rpm0, rpm1, de, dr = control*(Fossen.control_ub - Fossen.control_lb) - Fossen.control_lb

        # common subexpression elimination
        x0 = np.cos(psi)
        x1 = np.cos(theta)
        x2 = u*x1
        x3 = np.sin(phi)
        x4 = np.sin(psi)
        x5 = x3*x4
        x6 = np.sin(theta)
        x7 = np.cos(phi)
        x8 = x0*x7
        x9 = x4*x7
        x10 = x0*x3
        x11 = x1*x3
        x12 = x1*x7
        x13 = np.tan(theta)
        x14 = q*x3
        x15 = r*x7
        x16 = 1/x1
        x17 = Yvv*abs(v)
        x18 = m*xg
        x19 = p*x18
        x20 = m*zg
        x21 = r*x20
        x22 = m*u
        x23 = m*yg
        x24 = r*x23
        x25 = np.sin(dr)
        x26 = Kt0*rpm0
        x27 = Kt1*rpm1
        x28 = m*w
        x29 = -x28
        x30 = p*x23
        x31 = -B + W
        x32 = x3**2
        x33 = x6**2
        x34 = x1**2
        x35 = x7**2
        x36 = 1/(x32*x33 + x32*x34 + x33*x35 + x34*x35)
        x37 = x11*x36
        x38 = -p*(x29 - x30) - q*(x19 + x21) - r*(x22 - x24) - v*x17 + x25*(-x26 - x27) - x31*x37
        x39 = m**2
        x40 = xg**4
        x41 = x39*x40
        x42 = yg**2
        x43 = xg**2
        x44 = x39*x43
        x45 = x42*x44
        x46 = zg**2
        x47 = x44*x46
        x48 = Iyy*Izz
        x49 = Ixx*x48
        x50 = Ixx*Izz
        x51 = m*x50
        x52 = m*x48
        x53 = -x42*x52 - x43*x51 + x49
        x54 = Ixx*Iyy
        x55 = m*x54
        x56 = -x43*x55 - x46*x52
        x57 = Ixx*x41 + Iyy*x45 + Izz*x47 + x53 + x56
        x58 = yg**4
        x59 = x39*x58
        x60 = x42*x46
        x61 = x39*x60
        x62 = -x42*x55 - x46*x51
        x63 = Ixx*x45 + Iyy*x59 + Izz*x61 + x62
        x64 = zg**4
        x65 = x39*x64
        x66 = Ixx*x47 + Iyy*x61 + Izz*x65
        x67 = 1/(x57 + x63 + x66)
        x68 = x54*yg
        x69 = xg**3
        x70 = Ixx*x69
        x71 = yg**3
        x72 = Iyy*m
        x73 = x71*x72
        x74 = Izz*x46
        x75 = x23*x74
        x76 = x67*(x23*x70 - x68*xg + x73*xg + x75*xg)
        x77 = x50*zg
        x78 = zg**3
        x79 = Izz*x78
        x80 = x20*x42
        x81 = Iyy*x80
        x82 = x18*x79 + x20*x70 - x77*xg + x81*xg
        x83 = Zww*abs(w)
        x84 = m*v
        x85 = p*x20
        x86 = q*x23
        x87 = np.sin(de)
        x88 = np.cos(dr)
        x89 = x88*(x26 + x27)
        x90 = -x22
        x91 = q*x20
        x92 = x12*x36
        x93 = -p*(x84 - x85) - q*(x90 - x91) - r*(x19 + x86) - w*x83 - x31*x92 + x87*x89
        x94 = x67*x93
        x95 = Xuu*abs(u)
        x96 = q*x18
        x97 = np.cos(de)
        x98 = -x84
        x99 = r*x18
        x100 = x6/(x33 + x34)
        x101 = -p*(x21 + x86) - q*(x28 - x96) - r*(x98 - x99) - u*x95 + x100*x31 + x89*x97
        x102 = m**3
        x103 = Ixx*x102
        x104 = Iyy*x102
        x105 = Izz*x102
        x106 = x39*x42
        x107 = x39*x46
        x108 = x103*x43
        x109 = 1/(m*x49 + x103*x40 + x104*x42*x43 + x104*x58 + x104*x60 + x105*x43*x46 + x105*x60 + x105*x64 - x106*x48 - x106*x54 - x107*x48 - x107*x50 + x108*x42 + x108*x46 - x44*x50 - x44*x54)
        x110 = Ixx*x43
        x111 = x110*x23
        x112 = Iyy*x23
        x113 = -x111 - x112*x46 + x68 - x73
        x114 = p*q
        x115 = Qt0*rpm0
        x116 = Qt1*rpm1
        x117 = x88*(x115 + x116)
        x118 = -x19
        x119 = -x86
        x120 = x67*(B*(x100*yb + x37*xb) + Ixx*x114 - Iyy*x114 - Nrr*r*abs(r) - W*(x100*yg + x37*xg) - u*(x84 - x95*ycp + x99) - v*(x17*xcp + x24 + x90) - w*(x118 + x119) + x117*x87)
        x121 = Izz*m
        x122 = x121*x78
        x123 = Izz*x80 + x110*x20 + x122 - x77
        x124 = p*r
        x125 = -x21
        x126 = x67*(B*(-x100*zb - x92*xb) - Ixx*x124 + Izz*x124 - Mqq*q*abs(q) - W*(-x100*zg - x92*xg) - u*(x29 + x95*zcp + x96) - v*(x118 + x125) - w*(x22 - x83*xcp + x91) + x25*(-x115 - x116))
        x127 = x23*xg
        x128 = Izz*x127
        x129 = x128*zg
        x130 = x112*xg*zg
        x131 = x129 - x130
        x132 = q*r
        x133 = x67*(B*(-x37*zb + x92*yb) + Iyy*x132 - Izz*x132 - Kpp*p*abs(p) - W*(-x37*zg + x92*yg) - u*(x119 + x125) - v*(-x17*zcp + x28 + x30) - w*(x83*ycp + x85 + x98) + x117*x97)
        x134 = x48*zg
        x135 = Iyy*x20
        x136 = x111*zg - x134*yg + x135*x71 + x23*x79
        x137 = Ixx*zg
        x138 = x127*x137
        x139 = -x129 + x138
        x140 = Ixx*m
        x141 = x140*x69
        x142 = Ixx*x18
        x143 = x42*x72
        x144 = x141 + x142*x46 + x143*xg - x54*xg
        x145 = -Izz*x20*x43 - x122 + x134 - x81
        x146 = x101*x67
        x147 = x38*x67
        x148 = x130 - x138
        x149 = -x141 - x142*x42 - x18*x74 + x50*xg
        x150 = x112*x43 - x48*yg + x73 + x75
        x151 = x39*xg
        x152 = x39*x69
        x153 = -x135*xg + x151*x42*zg + x151*x78 + x152*zg
        x154 = -x128 + x151*x46*yg + x151*x71 + x152*yg
        x155 = -x121*x46 + x45
        x156 = -x143 + x47
        x157 = -x137*x23 + x39*x71*zg + x39*x78*yg + x44*yg*zg
        x158 = -x140*x43 + x61

        # dxdt
        return np.array([
            v*(x10*x6 - x9) + w*(x5 + x6*x8) + x0*x2,
            v*(x5*x6 + x8) + w*(-x10 + x6*x9) + x2*x4,
            -u*x6 + v*x11 + w*x12,
            p + x13*x14 + x13*x15,
            q*x7 - r*x3,
            x14*x16 + x15*x16,
            x101*x109*x57 + x113*x120 + x123*x126 + x131*x133 + x38*x76 + x82*x94,
            x101*x76 + x109*x38*(x53 + x63) + x120*x144 + x126*x139 + x133*x145 + x136*x94,
            x109*x93*(x49 + x56 + x62 + x66) + x120*x148 + x126*x149 + x133*x150 + x136*x147 + x146*x82,
            x120*x153 + x126*x154 + x131*x146 + x133*(-x121*x43 + x155 + x156 + x41 - x43*x72 + x48) + x145*x147 + x150*x94,
            x120*x157 + x123*x146 + x126*(-x121*x42 - x140*x42 + x155 + x158 + x50 + x59) + x133*x154 + x139*x147 + x149*x94,
            x113*x146 + x120*(-x140*x46 + x156 + x158 - x46*x72 + x54 + x65) + x126*x157 + x133*x153 + x144*x147 + x148*x94
        ])

    def plot(self, fname, states, controls=None, dpi=500):

        # create figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # plot positional trajectory
        ax.plot(states[:,0], states[:,1], states[:,2], 'k.-')

        # labels
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')
        ax.set_zlabel('$z$ [m]')
        
        # formating
        ax.grid('False')
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        # ax.set_aspect('equal')
        ax.auto_scale_xyz([-10, 10], [-10, 10], [-10, 10])

        # save
        fig.savefig(fname, bbox_inches='tight', dpi=500)
        plt.show()


if __name__ == '__main__':

    # instantiate Fossen model
    system = Fossen()

    # initial state
    state = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    
    # controller
    controller = lambda x: np.array([1000, 1000, 0.1, 0.1])

    # # propagate system
    t, x, u = system.propagate(state, controller, 0, 50, atol=1e-4, rtol=1e-4)

    # save
    system.plot('../img/trajectory.png', x, dpi=500)