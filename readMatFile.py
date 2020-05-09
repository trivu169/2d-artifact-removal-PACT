# Function code to read .mat file
# Author: Tri Vu
import scipy.io as sio
import numpy as np
import numpy.fft as nfft

def readReconsMat(name):
    data = sio.loadmat(name)
    data = data['p0_recons']
    return data

def readTrueMat(name):
    data = sio.loadmat(name)
    data = data['p0_true']
    return data

def readHrMat(name):
    data = sio.loadmat(name)
    data = data['p0_hr']
    return data

def readHilbertMat(name):
    data = sio.loadmat(name)
    data = data['p0_hil']
    return data

def getFFTReal(img):
    img = nfft.fft2(img)
    img = nfft.fftshift(img)
    return img.real

def getFFTImag(img):
    img = nfft.fft2(img)
    img = nfft.fftshift(img)
    return img.imag