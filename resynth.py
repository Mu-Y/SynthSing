# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 11:46:21 2018

@author: sharada

Synthesize from .npy files using WORLD

"""
import numpy as np
import scipy.fftpack as sfft
import pysptk
import pyworld as pw
import scipy.io.wavfile as wavfile
import os
import argparse


def mfsc_to_sp(mfsc, alpha=0.45, N=2048):
    mc = sfft.dct(mfsc, norm = 'ortho') # Mel cepstrum
    sp = pysptk.conversion.mc2sp(mc, alpha, N)  # Spectral envelope
    return sp


# Synthesizes from the .npy files in the folder
def synth(f0_dir, ap_dir, mfsc_dir):
    
    files = os.listdir(f0_dir)
    
    for file in files:
        # file_name = file.split('.')[0]
        file_name = '_'.join(file.split('_')[1:])  # Common file name
        
        # Get features for synthesis
        f0 = np.load(file)
        mfsc = np.load(mfsc_dir + '/mfsc_' + file_name)
        ap = np.load(ap_dir + '/ap_' + file_name)
        
        # Convert MFSC to SP
        sp = mfsc_to_sp(mfsc)
        # Synthesize the audio
        _synth(file_name, f0, ap, sp)
    
    print('Finished synthesis')
        
def _synth(file, f0, ap, sp, fs=32000, fft_size=2048):      
    file_name = file.split('.')[0] + '.wav'
    y =  pw.synthesize(f0, sp, ap, fs, pw.default_frame_period)
    wavfile.write(file_name, fs, y)

if __name__ == '__main__':
    fs = 32000
    fft_size = 2048

    p = argparse.ArgumentParser()
    p.add_argument('--mfsc_file', type=str)
    p.add_argument('--f0_file', type=str)
    p.add_argument('--ap_file', type=str)
    p.add_argument('--wav_out_path', type=str)

    args = p.parse_args()

    f0 = np.load(args.f0_file)

    ap = np.load(args.ap_file)
    ap = ap[20:]
    ap = pw.decode_aperiodicity(ap, fs, fft_size)

    mfsc = np.load(args.mfsc_file)
    mfsc = mfsc[20:]

    sp = mfsc_to_sp(mfsc)

    y =  pw.synthesize(f0[:1770], sp[:1770], ap[:1770], fs, pw.default_frame_period)
    y /= np.max(np.abs(y))
    wavfile.write(args.wav_out_path, fs, y)


