from types import MethodType, FunctionType
from scipy.integrate import ode, odeint, simps
from scipy import linalg
from scipy.optimize import nnls
from scipy.linalg import expm, norm
import matplotlib.pyplot as plt
import numpy as np
from nnls_ODE import nnl_ode
import pickle

class KineticsProp:
    """
    Propagation of kinetic equations
    """

    def __init__(self, **kwargs):
        """
         The following parameters are to be specified as arguments:

               pump_central -- central wavelength of pump pulse in nm
               pump_bw -- pump bandwidth in nm
               dump_central -- central wavelength of dump pulse in nm
               dump_bw -- dump bandwidth in nm
        """

        # Save all attributes
        for name, value in kwargs.items():
            # If the value supplied is a function,
            # then dynamically assign it as a method,
            # otherwise bind it a property.
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        # ===========================================================================#
        # ------------------- READ SPECTRAL PARAMETERS FROM FILE --------------------#
        # ===========================================================================#

        # data = np.loadtxt("Cph8_RefSpectra.csv", delimiter=',')
        data = np.loadtxt("Cph8_RefCrossSect.csv", delimiter=',')

        self.lamb = np.array(data[:, 0])
        self.Pr_abs = np.array(data[:, 1])
        self.Pr_ems = np.array(data[:, 3])
        self.Pfr_abs = np.array(data[:, 2])
        self.Pfr_ems = np.array(data[:, 4])

        self.beam_area = np.pi * self.beam_diameter ** 2 / 4.

    def I_pump(self, t):
        return np.exp(-((t - self.t0_pump) / self.pump_width) ** 2)

    def I_dump(self, t):
        return np.exp(-((t - self.t0_dump) / self.dump_width) ** 2)

    def propagate(self, p0):
        """
        Propagate a state with withe initial condition p0
        y(t_final) from y(t)
        """

        def M(t):
            """
            Return matrix of the rate equations
            """
            tmp = self.V_pump * self.I_pump(t)
            tmp += self.V_dump * self.I_dump(t)
            tmp += self.G0
            return tmp

        solver = nnl_ode(M).set_initial_value(p0)

        # p = np.array(
        #     [solver.integrate(tau) for tau in self.t_axis]
        # ).T

        p = solver.integrate(self.t_axis[-1]).T
        print p.sum(axis=0)

        return p

    def __call__(self, parameters):
        pump_energy, dump_energy, self.pump_width, self.dump_width, self.t0_pump, self.t0_dump = parameters

        pump_spectra = 1. / (np.sqrt(np.pi) * self.pump_bw) \
            * np.exp(-((self.lamb - self.pump_central) / self.pump_bw) ** 2)

        dump_spectra = 1. / (np.sqrt(np.pi) * self.dump_bw) \
            * np.exp(-((self.lamb - self.dump_central) / self.dump_bw) ** 2)

        self.t_axis = t_axis = np.linspace(0., self.T_max, self.T_steps)

        I_pump = self.I_pump(t_axis)
        I_dump = self.I_dump(t_axis)

        scale_pump = 1.e5 * pump_energy / simps(I_pump, t_axis)  # ASK ZAK
        scale_dump = 1.e5 * dump_energy / simps(I_dump, t_axis)

        pump_spectra *= scale_pump
        dump_spectra *= scale_dump

        scale_factor = 5030.68719187041 *1e9
        K_12_pump = simps(scale_factor * pump_spectra * self.lamb * self.Pr_abs / self.beam_area, self.lamb)
        K_12_dump = simps(scale_factor * dump_spectra * self.lamb * self.Pr_abs / self.beam_area, self.lamb)

        K_34_pump = simps(scale_factor * pump_spectra * self.lamb * self.Pr_ems / self.beam_area, self.lamb)
        K_34_dump = simps(scale_factor * dump_spectra * self.lamb * self.Pr_ems / self.beam_area, self.lamb)

        K_67_pump = simps(scale_factor * pump_spectra * self.lamb * self.Pfr_abs / self.beam_area, self.lamb)
        K_67_dump = simps(scale_factor * dump_spectra * self.lamb * self.Pfr_abs / self.beam_area, self.lamb)

        K_89_pump = simps(scale_factor * pump_spectra * self.lamb * self.Pfr_ems / self.beam_area, self.lamb)
        K_89_dump = simps(scale_factor * dump_spectra * self.lamb * self.Pfr_ems / self.beam_area, self.lamb)


        ###############################################################################

        self.G0 = np.zeros([10, 10])

        self.G0[0, 3] = self.A_41
        self.G0[0, 9] = self.A_101
        self.G0[1, 1] = -self.A_23
        self.G0[2, 1] = self.A_23
        self.G0[2, 2] = -self.A_34-self.A_35
        self.G0[3, 2] = self.A_34
        self.G0[3, 3] = -self.A_41
        self.G0[4, 2] = self.A_35
        self.G0[4, 4] = -self.A_56
        self.G0[5, 4] = self.A_56
        self.G0[5, 8] = self.A_96
        self.G0[6, 6] = -self.A_78
        self.G0[7, 6] = self.A_78
        self.G0[7, 7] = -self.A_89 - self.A_810
        self.G0[8, 7] = self.A_89
        self.G0[8, 8] = -self.A_96
        self.G0[9, 7] = self.A_810
        self.G0[9, 9] = -self.A_101

        self.V_pump = np.zeros_like(self.G0)
        self.V_dump = np.zeros_like(self.G0)

        self.V_pump[0, 1] = self.V_pump[1, 0] = K_12_pump
        self.V_pump[0, 0] = self.V_pump[1, 1] = -K_12_pump
        self.V_pump[2, 2] = self.V_pump[3, 3] = -K_34_pump
        self.V_pump[2, 3] = self.V_pump[3, 2] = K_34_pump

        self.V_pump[5, 6] = self.V_pump[6, 5] = K_67_pump
        self.V_pump[5, 5] = self.V_pump[6, 6] = -K_67_pump
        self.V_pump[7, 7] = self.V_pump[8, 8] = -K_89_pump
        self.V_pump[7, 8] = self.V_pump[8, 7] = K_89_pump

        self.V_dump[0, 1] = self.V_dump[1, 0] = K_12_dump
        self.V_dump[0, 0] = self.V_dump[1, 1] = -K_12_dump
        self.V_dump[2, 2] = self.V_dump[3, 3] = -K_34_dump
        self.V_dump[2, 3] = self.V_dump[3, 2] = K_34_dump
        self.V_dump[5, 6] = self.V_dump[6, 5] = K_67_dump
        self.V_dump[5, 5] = self.V_dump[6, 6] = -K_67_dump
        self.V_dump[7, 7] = self.V_dump[8, 8] = -K_89_dump
        self.V_dump[7, 8] = self.V_dump[8, 7] = K_89_dump

        ####################ss###########################################################
        #
        #   Transfer matrix construction
        #
        ###############################################################################
        # M = np.transpose(
        #     [self.propagate(e) for e in np.eye(10)]
        # )
        #
        # vals, vecs = linalg.eig(M)
        #
        # v = vecs[:, np.abs(vals - 1).argmin()] # eigenvector corresponding to eigenvalue 1, eq. state

        e = (0., 0., 0., 0., 0., 1., 0., 0., 0., 0.)
        return self.propagate(e)

        # with open("eigenvalues1.pickle", "wb") as output_file:
        #     pickle.dump(M, output_file)
        # return v.real / v.real.sum()             # 5th element corresponds to FR1 state population

####################################################################################################
#                                                                                                  #
#           CLASS FOR SOLVING NON-NEGATIVE ODE USING NON-NEGATIVE LEAST SQUARES IN PYTHON          #
#                                                                                                  #
####################################################################################################

if __name__=='__main__':

    print(KineticsProp.__doc__)

    print KineticsProp(
        # Pulses characterization
        pump_central=625.,
        pump_bw=40.,

        dump_central=835.,
        dump_bw=10.,

        beam_diameter=200.,

        T_max=100.,
        T_steps=1000,

        A_41=1 / .150,
        A_23=1 / .150,
        A_35=1 / 68.5,
        A_34=2.5 / 68.5,
        A_56=1 / 0.01,

        A_96=1 / .050,
        A_78=1 / .050,
        A_810=1 / 2.5,
        A_89=2.0,
        A_101=1 / 0.01,

        Iterations=51,
    )(
        (0.25, 2.0, .100, .150, 0.5, 0.5251)
    )