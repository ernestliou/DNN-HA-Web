import os
import sys
import time
import numpy as np
import scipy.signal as sp_sig
import scipy.io.wavfile as sio_wav

# Ensure the root project directory is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
    
sys_dir = os.path.join(project_root, "sys")
if sys_dir not in sys.path:
    sys.path.append(sys_dir)

# 強制設定 TensorFlow 使用舊版 Keras 引擎
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from tensorflow.keras.models import model_from_json

from extra_functions import wavfile_read, rms, slice_1dsignal, reconstruct_wav
from gui.i18n import get_i18n

# 固定參數
FS_MODEL = 20e3
NENC = 6
P0 = 2e-5
OVERLAP = 0
MODEL_DIR = 'CNN-HA-12layers'
FREQS = [125, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000]

# 載入模型 (全域載入，避免每次處理都重新讀取)
print("Loading DNN-HA model...")
with open(os.path.join(MODEL_DIR, "Gmodel.json"), "r") as json_file:
    loaded_model_json = json_file.read()
modelp = model_from_json(loaded_model_json)
modelp.load_weights(os.path.join(MODEL_DIR, "Gmodel.h5"))
print("Model loading complete.")

# 快取濾波器係數以利即時處理
_taps_cache = {}

def get_simulation_taps(fs, audiogram_freqs, audiogram_loss):
    key = (fs, tuple(audiogram_freqs), tuple(audiogram_loss))
    if key in _taps_cache:
        return _taps_cache[key]
    
    nyq = fs / 2
    freqs = [0.0] + list(audiogram_freqs) + [nyq]
    gains_dB = [-audiogram_loss[0]] + [-loss for loss in audiogram_loss] + [-audiogram_loss[-1]]
    gains_linear = [10**(g/20.0) for g in gains_dB]
    
    normalized_freqs = [f / nyq for f in freqs]
    
    valid_freqs = []
    valid_gains = []
    for f, g in zip(normalized_freqs, gains_linear):
        if f <= 1.0:
            if valid_freqs and f <= valid_freqs[-1]:
                continue
            valid_freqs.append(f)
            valid_gains.append(g)
            
    if valid_freqs[-1] < 1.0:
        valid_freqs.append(1.0)
        valid_gains.append(valid_gains[-1])
        
    numtaps = 1025
    taps = sp_sig.firwin2(numtaps, valid_freqs, valid_gains)
    _taps_cache[key] = taps
    return taps

def simulate_hearing_loss(audio_data, fs, audiogram_freqs, audiogram_loss):
    """
    透過 FIR 濾波器模擬聽力受損
    """
    taps = get_simulation_taps(fs, audiogram_freqs, audiogram_loss)
    simulated_audio = sp_sig.convolve(audio_data, taps, mode='same')
    return simulated_audio

def stream_process(audio, mode, l_val, sim_gain_db, *audiogram_args):
    """
    即時串流處理函式
    """
    if audio is None:
        return None
        
    fs, y = audio
    
    # 轉換為 float32 並正規化
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    elif y.dtype == np.int32:
        y = y.astype(np.float32) / 2147483648.0
    else:
        y = y.astype(np.float32)
        
    # 單聲道處理
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
        
    # 重採樣至 20kHz (模型要求)
    if fs != FS_MODEL:
        y = sp_sig.resample_poly(y, int(FS_MODEL), int(fs))
        
    L = float(l_val)
    stim = y * (P0 * 10**(L/20)) / 0.01
    
    if mode == 0: # Original
        out_y = stim
    elif mode == 1: # Simulated
        out_y = simulate_hearing_loss(stim, FS_MODEL, FREQS, list(audiogram_args))
        # 套用額外增益
        out_y = out_y * (10**(float(sim_gain_db)/20))
    else: # DNN-HA 補償
        stim_len = len(stim)
        rem = stim_len % 64
        if rem:
            stim_padded = np.pad(stim, (0, 64 - rem))
        else:
            stim_padded = stim
            
        stim_input = stim_padded.reshape(1, -1, 1)
        audiogram_input = np.array(audiogram_args).reshape(1, 1, 9)
        audiogram_rep = np.tile(audiogram_input, (1, stim_input.shape[1]//64, 1))
        
        # 執行推論
        stimp = modelp.predict([stim_input, audiogram_rep], verbose=0)
        out_y = stimp[0, :, 0]
        
        if rem:
            out_y = out_y[:stim_len]
            
        max_out = np.max(np.abs(out_y))
        if max_out > 0:
            out_y = (out_y / max_out) * 0.1 # 即時模式音量略低以防爆音
            
    # 轉回 int16
    out_y_int = np.clip(out_y * 32767, -32768, 32767).astype(np.int16)
    return (int(FS_MODEL), out_y_int), out_y

def process_audio(wavfile_input, l_val, snr_str, frame_size_val, sim_gain_db, *audiogram_args):
    """
    Gradio 的主要處理邏輯
    """
    t_msg = get_i18n()
    if not wavfile_input:
        return None, None, t_msg.get("err_no_audio"), None, None, None
        
    try:
        audiogram_input = list(audiogram_args)
        L = float(l_val)
        snr_str = str(snr_str).strip()
        SNR = float(snr_str) if snr_str else ''
        frame_size = int(frame_size_val)
        
        if frame_size > 0:
            if frame_size < 64:
                return None, None, t_msg.get("err_frame_size_small"), None, None, None
            if frame_size % 64 != 0:
                return None, None, t_msg.get("err_frame_size_mul"), None, None, None
        
        # --- 音訊前處理 ---
        speechsignal, fs_speech = wavfile_read(wavfile_input) 
        
        if len(speechsignal.shape) > 1 and speechsignal.shape[1] > 1:
            speechsignal = np.mean(speechsignal, axis=1)
            
        if FS_MODEL != fs_speech :
            stim_full = sp_sig.resample_poly(speechsignal, FS_MODEL, fs_speech)
        else :
            stim_full = speechsignal

        stim_length = stim_full.size
        rem = stim_length % (2**NENC)
        if rem:
            stim_length = stim_length + int(2**NENC-rem)

        stim = np.zeros((1,stim_length))
        stim[0,:stim_full.size] = P0 * 10**(L/20) * stim_full / rms(stim_full)
        
        raw_original = stim[0, :].copy()

        if SNR != '':
            noise = np.random.normal(size=stim_length)
            speechrms = rms(stim)
            noiserms = rms(noise)
            ratio = (speechrms/noiserms) * np.sqrt(10**(-SNR/10))
            stim[0,:] = stim[0,:] + ratio*noise

        stim = np.expand_dims(stim, axis=2)
        
        if frame_size:
            stim_cropped = slice_1dsignal(stim[0,:,0], frame_size, 0, stride=1-OVERLAP)
            stim_cropped = np.reshape(stim_cropped, (-1,frame_size,1))
        else:
            stim_cropped = stim
            
        audiogram_rep = np.tile(audiogram_input, (stim_cropped.shape[0], int(stim_cropped.shape[1]/(2**NENC)), 1))
        
        t = time.time()
        stimp_cropped = modelp.predict([stim_cropped, audiogram_rep])
        proc_time = time.time() - t
        
        if frame_size:
            stimp_cropped = reconstruct_wav(stimp_cropped[:,:,0], stride_factor=1-OVERLAP)
            
        stimp = np.zeros((1, stim.shape[0], stim.shape[1], stim.shape[2]))
        stimp[0,0] = np.reshape(stimp_cropped[:,:stim.shape[1]], (1,-1,1))
        
        raw_audio_for_sim = stim[0,:,0]
        reference_audio = raw_audio_for_sim * (0.1 / (P0 * 10**(70/20)))
            
        simulated_audio_float = simulate_hearing_loss(reference_audio, FS_MODEL, FREQS, audiogram_input)
        simulated_audio_float = simulated_audio_float * (10**(float(sim_gain_db)/20))
        
        save_wav_dir = 'wavfiles'
        if not os.path.exists(save_wav_dir):
            os.makedirs(save_wav_dir)
            
        base_name = os.path.basename(wavfile_input)
        if base_name.endswith('.wav'):
            base_name = base_name[:-4]
            
        out_filename = os.path.join(save_wav_dir, f"{base_name}_WEB_processed_20k.wav")
        raw_out_audio = stimp[0,0,:,0]
        max_out = np.max(np.abs(raw_out_audio))
        if max_out > 0:
            norm_audio = (raw_out_audio / max_out) * 0.8
        else:
            norm_audio = raw_out_audio
        audio_data = np.clip(norm_audio * 32767, -32768, 32767).astype(np.int16)
        sio_wav.write(out_filename, int(FS_MODEL), audio_data)
        
        sim_filename = os.path.join(save_wav_dir, f"{base_name}_WEB_simulated_20k.wav")
        sim_audio_data = np.clip(simulated_audio_float * 32767, -32768, 32767).astype(np.int16)
        sio_wav.write(sim_filename, int(FS_MODEL), sim_audio_data)
        
        out_spl = 20*np.log10(rms(stimp[0,0],axis=None))-20*np.log10(P0)
        status_msg = t_msg.get("msg_proc_done", "").format(time=proc_time, spl=out_spl)
        
        return out_filename, sim_filename, status_msg, raw_original, simulated_audio_float, norm_audio
        
    except Exception as e:
        import traceback
        err_str = f"{str(e)}\n{traceback.format_exc()}"
        return None, None, t_msg.get("msg_err_general", "").format(err=err_str), None, None, None
