import numpy as np


def nu(mach, gamma):
    a = np.sqrt((gamma + 1) / (gamma - 1))
    b = np.sqrt((gamma - 1) / (gamma + 1) * (mach**2 - 1))
    c = np.sqrt(mach**2 - 1)
    return a * np.arctan(b) - np.arctan(c)


def temp_2(temp_1, mach_1, mach_2, gamma):
    num = 1 + (gamma - 1) / 2 * mach_1**2
    den = 1 + (gamma - 1) / 2 * mach_2**2
    return temp_1 * num / den


def pressure_2(pressure_1, mach_1, mach_2, gamma):
    num = 1 + (gamma - 1) / 2 * mach_1**2
    den = 1 + (gamma - 1) / 2 * mach_2**2
    return (pressure_1 * num / den)**(gamma / (gamma - 1))


def rho_2(rho_1, mach_1, mach_2, gamma):
    num = 1 + (gamma - 1) / 2 * mach_1**2
    den = 1 + (gamma - 1) / 2 * mach_2**2
    return (rho_1 * num / den)**(1 / (gamma - 1))
