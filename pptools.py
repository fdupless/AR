import time, sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if not sys.version_info[0] < 3:
    from importlib import reload
import scipy.signal as scisig
from scipy.interpolate import UnivariateSpline as uspl


def fft(func):
    fftfunc = np.fft.rfft(func)
    return np.real(fftfunc) / len(func), np.imag(fftfunc) / len(func)


from scipy.interpolate import LSQUnivariateSpline as uspl


def ffttospline(dataraw, ploton=False, knots=[10, 25]):
    knots = [10, 25]
    degsp = 1
    fftdata = fft(dataraw)

    fftdata2 = np.zeros(len(fftdata[0]) + len(fftdata[1]))
    for i in range(len(fftdata[0])):
        fftdata2[2 * i] = abs(fftdata[0][i])
        fftdata2[2 * i + 1] = abs(fftdata[1][i])

    constterms = np.array([fftdata[0][0]])
    fftdata3 = scisig.savgol_filter(fftdata2[2:], 5, 0)

    x = np.linspace(1, len(fftdata3), len(fftdata3))
    fftsplinedparams = uspl(x, fftdata3, knots, k=degsp).get_coeffs()

    feats = np.append(constterms, fftsplinedparams)

    if ploton:

        fs = 15
        x2 = np.linspace(1, len(fftdata[0]), len(fftdata[0]) - 1)
        plt.figure(figsize=(5, 8))

        ax2 = plt.subplot(3, 1, 1)
        plt.plot(x, dataraw)
        ax2.set_ylabel(r"$a_{ankle}$ $[m/s^2]$", fontsize=fs)
        ax2.set_xlabel("Sample Number", fontsize=fs)

        ax1 = plt.subplot(3, 1, 2)
        plt.plot(x2, fftdata[0][1:], label="Real", color="blue")
        plt.plot(x2, fftdata[1][1:], label="Imag", color="red")
        plt.legend(loc="upper right", ncol=2, fancybox=True)
        ax1.set_xlabel("DFT mode number", fontsize=fs)
        ax1.set_ylabel(r"$DFT(a_{ankle})$ $[m/s^2]$", fontsize=fs)
        ax1.get_legend()

        usp = uspl(x, fftdata3, knots, k=degsp)
        splinefit = [usp(i) for i in x]

        ax3 = plt.subplot(3, 1, 3)
        plt.plot(
            x, fftdata2[2:], color="green", label=r"|Real|$_{even}$+|Imag|$_{odd}$"
        )
        plt.plot(x, splinefit, color="black", label="Spline Fit")
        ax3.set_xlabel("DFT mode number", fontsize=fs)
        ax3.set_ylabel(r"$DFT(a_{ankle})$ $[m/s^2]$", fontsize=fs)
        plt.legend(loc="upper right", ncol=2, fancybox=True)
        ax3.get_legend()

        plt.tight_layout()
        plt.savefig("fourier")
        plt.close()
    return feats
