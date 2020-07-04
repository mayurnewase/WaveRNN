import torch
from models.fatchord_version import WaveRNN
from utils import hparams as hp
from utils.text.symbols import symbols
from utils.paths import Paths
from models.tacotron import Tacotron
import argparse
from utils.text import text_to_sequence
from utils.display import save_attention, simple_table
from utils.dsp import reconstruct_waveform, save_wav
import numpy as np
from utils.display import save_spectrogram
import pickle
import gc

input_text = "hey there"
save_attn = True
vocoder = "wavernn"
pad_input=False
max_len = 90
hp_file = "hparams.py"

tts_weights = None
voc_weights = None

batched =True
force_cpu = True


def lambda_handler(e,c):
    hp.configure(hp_file)  # Load hparams from file
    if vocoder == 'wavernn':
        target = hp.voc_target
        overlap = hp.voc_overlap
        batched = hp.voc_gen_batched

    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)
    if not force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print('\nInitialising WaveRNN Model...\n')
    # Instantiate WaveRNN Model
    voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                        fc_dims=hp.voc_fc_dims,
                        bits=hp.bits,
                        pad=hp.voc_pad,
                        upsample_factors=hp.voc_upsample_factors,
                        feat_dims=hp.num_mels,
                        compute_dims=hp.voc_compute_dims,
                        res_out_dims=hp.voc_res_out_dims,
                        res_blocks=hp.voc_res_blocks,
                        hop_length=hp.hop_length,
                        sample_rate=hp.sample_rate,
                        mode=hp.voc_mode).to(device)

    voc_load_path = voc_weights if voc_weights else paths.voc_latest_weights
    voc_model.load(voc_load_path)

    print('\nInitialising Tacotron Model...\n')
    # Instantiate Tacotron Model
    tts_model = Tacotron(embed_dims=hp.tts_embed_dims,
                         num_chars=len(symbols),
                         encoder_dims=hp.tts_encoder_dims,
                         decoder_dims=hp.tts_decoder_dims,
                         n_mels=hp.num_mels,
                         fft_bins=hp.num_mels,
                         postnet_dims=hp.tts_postnet_dims,
                         encoder_K=hp.tts_encoder_K,
                         lstm_dims=hp.tts_lstm_dims,
                         postnet_K=hp.tts_postnet_K,
                         num_highways=hp.tts_num_highways,
                         dropout=hp.tts_dropout,
                         stop_threshold=hp.tts_stop_threshold).to(device)

    tts_load_path = tts_weights if tts_weights else paths.tts_latest_weights
    tts_model.load(tts_load_path)

    inputs = [text_to_sequence(input_text.strip(), hp.tts_cleaner_names)]
    
    print("after sequenceing ", inputs)
    def pad1d(x, max_len):
      return np.pad(x, (0, max_len - len(x)), mode='constant')

    if pad_input:
      print(">>>padding input")
      inputs = [pad1d(inputs[0], max_len)]
    print("cleaned input text is ", inputs)

    voc_k = voc_model.get_step() // 1000
    tts_k = tts_model.get_step() // 1000

    simple_table([('Tacotron', str(tts_k) + 'k'),
                ('r', tts_model.r),
                ('Vocoder Type', 'WaveRNN'),
                ('WaveRNN', str(voc_k) + 'k'),
                ('Generation Mode', 'Batched' if batched else 'Unbatched'),
                ('Target Samples', target if batched else 'N/A'),
                ('Overlap Samples', overlap if batched else 'N/A')])


    for i, x in enumerate(inputs, 1):

        print(f'\n| Generating {i}/{len(inputs)}')
        _, m, attention = tts_model.generate(x)

        # Fix mel spectrogram scaling to be from 0 to 1
        m = (m + 4) / 8
        np.clip(m, 0, 1, out=m)
        print("mel spectrogram shape is ",m.shape)
        
        del tts_model
        gc.collect()

        if vocoder == 'wavernn' and batched:
            v_type = 'wavernn_batched'
        else:
            v_type = 'wavernn_unbatched'

        save_path = paths.tts_output/f'__input_{input_text[:10]}_{v_type}_{tts_k}k.wav'
        
        if vocoder == 'wavernn':
            m = torch.tensor(m).unsqueeze(0)
            voc_model.generate(m, save_path, batched, hp.voc_target, hp.voc_overlap, hp.mu_law)

    print('\n\nDone.\n')
