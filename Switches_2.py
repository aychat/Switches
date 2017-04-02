import numpy as np
import pickle
# from kinetic_model_grid import KineticsProp
from kinetics_old import KineticsProp
from numpy.random import uniform
from multiprocessing import Pool
from itertools import product
import time

switch_1 = dict(
        # Pulses characterization
        pump_central=625.,
        pump_bw=30.,

        dump_central=835.,
        dump_bw=10.,

        beam_diameter=200.,

        T_max=50.0,
        T_steps=100,

        A_41=1. / 0.15,
        A_23=1. / 0.15,
        A_35=1. / 61.0,
        A_34=1. / 26.0,
        A_56=1. / .001,

        A_96=1. / .010,
        A_78=1. / .050,
        A_810=1. / 1.50,
        A_89=1. / 0.30,
        A_101=1 / .001,

    )

switch_2 = dict(
        # Pulses characterization
        pump_central=625.,
        pump_bw=30.,

        dump_central=835.,
        dump_bw=10.,

        beam_diameter=200.,

        T_max=50.0,
        T_steps=100,

        A_41=1. / 0.12,
        A_23=1. / 0.12,
        A_35=1. / 71.0,
        A_34=1. / 30.0,
        A_56=1. / .001,

        A_96=1. / .015,
        A_78=1. / .060,
        A_810=1. / 1.25,
        A_89=1. / 0.35,
        A_101=1 / .001,

    )

n = 1000

pump = uniform(0.0, 2.5, n)
dump = uniform(0.0, 2.5, n)
pump_width = uniform(.05, .35, n)
dump_width = uniform(.05, .35, n)
delay = uniform(.005, .105, n)


def two_switches(params):
    print params
    return KineticsProp(**switch_1)(params), KineticsProp(**switch_2)(params)


if __name__ == '__main__':
    start = time.time()
    result = map(
        two_switches, zip(pump, dump, pump_width, dump_width, delay)
    )
    end = time.time()
    time = end - start

    with open('result_sw2_rand_local.pickle', 'wb') as file_out:
        pickle.dump(
            {
                'switch_params_1': switch_1,
                'switch_params_2': switch_2,
                'params': zip(pump, dump, pump_width, dump_width, delay),
                'result': result,
                'time': time
            },
            file_out
        )