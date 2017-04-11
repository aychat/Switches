from types import MethodType, FunctionType
from scipy.integrate import ode, odeint, simps
from scipy import linalg
import numpy as np
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

    def propagate(self, G0, V_pump, V_dump, p0):
        """
        Propagate a state with withe initial condition p0
        """

        def jac(p, t):
            """
            Return Jacobian of the rate equations
            """
            tmp = V_pump * self.I_pump(t)
            tmp += V_dump * self.I_dump(t)
            tmp += G0
            return tmp

        def rhs(p, t):
            """
            Return the r.h.s. of the rate equations
            """
            return jac(p, t).dot(p)

        res = odeint(rhs, p0, self.t_axis, Dfun=jac)
        # print res[-1]
        return res

    def __call__(self, parameters):
        pump_energy, dump_energy, self.pump_width, self.dump_width, self.pump_dump_delay = parameters
        self.t0_pump = 0.5
        self.t0_dump = self.t0_pump + self.pump_dump_delay

        pump_spectra = 1. / (np.sqrt(np.pi) * self.pump_bw) \
            * np.exp(-((self.lamb - self.pump_central) / self.pump_bw) ** 2)

        dump_spectra = 1. / (np.sqrt(np.pi) * self.dump_bw) \
            * np.exp(-((self.lamb - self.dump_central) / self.dump_bw) ** 2)

        self.t_axis = t_axis = np.linspace(0., self.T_max, self.T_steps)

        I_pump = self.I_pump(t_axis)
        I_dump = self.I_dump(t_axis)

        scale_pump = 1.e5 * pump_energy / simps(I_pump, t_axis) # ASK ZAK
        scale_dump = 1.e5 * dump_energy / simps(I_dump, t_axis)

        pump_spectra *= scale_pump
        dump_spectra *= scale_dump

        scale_factor = 5030.68719187041 * 1.e9

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

        ###############################################################################
        #
        #   Transfer matrix construction
        #
        ###############################################################################
        M = np.asarray(
            [self.propagate(self.G0, self.V_pump, self.V_dump, e)[-1] for e in np.eye(10)]
        )

        with open("M1.pickle", "wb") as f:
            pickle.dump(M, f)

        e = (0.5, 0., 0., 0., 0., 0.5, 0., 0., 0., 0.)
        value = self.propagate(self.G0, self.V_pump, self.V_dump, e)[-1]

        # with open("kinetic_dyn.pickle", "wb") as f:
        #     pickle.dump(value.T, f)

        np.set_printoptions(precision=4, suppress=True)

        vals, vecs = linalg.eig(M.T)
        v = vecs[:, np.abs(vals - 1).argmin()]
        result = v.real / v.real.sum()
        return result[0], result[5]
####################################################################################################
#
#   Example
#
####################################################################################################


if __name__=='__main__':

    print(KineticsProp.__doc__)

    print KineticsProp(
        # Pulses characterization
        pump_central=625.,
        pump_bw=30.,

        dump_central=835.,
        dump_bw=10.,

        beam_diameter=200.,

        T_max=60.,
        T_steps=60000,

        A_41=1. / 0.15,
        A_23=1. / 0.15,
        A_35=1. / 61.0,
        A_34=1. / 26.0,
        A_56=1. / 1.,

        A_96=1. / .010,
        A_78=1. / .050,
        A_810=1. / 1.50,
        A_89=1. / 0.30,
        A_101=1. / 1.,

        Iterations=51,
    )(
        (0.0, 0.0, .025, .150, 0.1051)
    )