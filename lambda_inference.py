#import torch
from models.fatchord_version import WaveRNN
from utils import hparams as hp
#from utils.text.symbols import symbols
#from utils.paths import Paths
#from models.tacotron import Tacotron
import argparse
#from utils.text import text_to_sequence
##from utils.display import save_attention, simple_table
#from utils.dsp import reconstruct_waveform, save_wav
#import numpy as np
#from utils.display import save_spectrogram
#import pickle

def lambda_handler(e,c):
    print("hey")