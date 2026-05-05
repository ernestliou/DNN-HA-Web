"""
This file (web_app.py) contains modifications and a Web UI wrapper developed by CHAO-CHIA, LIU.
Copyright (c) 2026 CHAO-CHIA, LIU. All rights reserved for the Web UI components.

The underlying core audio processing logic and deep neural network models are 
based on and derived from the original DNN-HA project (https://github.com/fotisdr/DNN-HA).

REQUIRED UGent ACADEMIC LICENSE NOTICE:
© copyright 2020 Ghent University – Universiteit Gent, all rights reserved; this Derivative work is made available for non-commercial academic research purposes and subject to an UGent Academic License (https://github.com/fotisdr/DNN-HA/blob/main/LICENSE.txt)
"""
import os
import sys

# 強制設定 TensorFlow 使用舊版 Keras 引擎
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import gradio as gr
import numpy as np
import scipy.signal as sp_sig
import scipy.io.wavfile as sio_wav
import time
import json
import locale

# 延遲匯入 TF 以免影響啟動速度，但在這裡先匯入也無妨
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt

# 嘗試匯入 PyOctaveBand
try:
    import PyOctaveBand
except ImportError:
    PyOctaveBand = None

# 匯入 sys 目錄中的共用函式
from extra_functions import wavfile_read, rms, slice_1dsignal, reconstruct_wav

# --- 多語系 (i18n) 支援 ---
LOCALES_DIR = "locale"
translations = {}

for lang_code in ["zh_TW", "en_US"]:
    loc_file = os.path.join(LOCALES_DIR, f"{lang_code}.json")
    if os.path.exists(loc_file):
        with open(loc_file, "r", encoding="utf-8") as f:
            translations[lang_code] = json.load(f)

if "en_US" not in translations:
    translations["en_US"] = {} # Fallback

try:
    sys_lang, _ = locale.getdefaultlocale()
    if sys_lang and sys_lang in translations:
        default_lang = sys_lang
    else:
        default_lang = "en_US"
except:
    default_lang = "en_US"

def get_i18n(lang):
    return translations.get(lang, translations.get("en_US", {}))

# 固定參數 (與原腳本相同)
FS_MODEL = 20e3
NENC = 6
P0 = 2e-5
OVERLAP = 0
MODEL_DIR = 'CNN-HA-12layers'
WAV_DBREF = 110

# 載入模型 (全域載入，避免每次處理都重新讀取)
print("Loading DNN-HA model...")
with open(os.path.join(MODEL_DIR, "Gmodel.json"), "r") as json_file:
    loaded_model_json = json_file.read()
modelp = model_from_json(loaded_model_json)
modelp.load_weights(os.path.join(MODEL_DIR, "Gmodel.h5"))
print("Model loading complete.")

FREQS = [125, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000]

def simulate_hearing_loss(audio_data, fs, audiogram_freqs, audiogram_loss):
    """
    透過 FIR 濾波器模擬聽力受損
    """
    taps = get_simulation_taps(fs, audiogram_freqs, audiogram_loss)
    simulated_audio = sp_sig.convolve(audio_data, taps, mode='same')
    return simulated_audio

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
    
    # mode is now index 0, 1, 2
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
    t_msg = get_i18n(default_lang)
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

# --- Gradio UI ---
AUDIOGRAM_FILE = os.path.join("gui", "audiogram.json")
default_audiogram = [10, 15, 20, 25, 30, 40, 45, 50, 55]

if os.path.exists(AUDIOGRAM_FILE):
    try:
        with open(AUDIOGRAM_FILE, "r") as f:
            saved_audiogram = json.load(f)
            if isinstance(saved_audiogram, list) and len(saved_audiogram) == 9:
                default_audiogram = saved_audiogram
    except Exception as e:
        print(f"無法載入 {AUDIOGRAM_FILE}: {e}")

def save_audiogram_to_file(*args):
    t_msg = get_i18n(default_lang)
    try:
        with open(AUDIOGRAM_FILE, "w") as f:
            json.dump(list(args), f)
        return t_msg.get("msg_audio_saved", "")
    except Exception as e:
        return t_msg.get("msg_audio_save_err", "").format(err=str(e))

def load_audiogram_from_file(file):
    t_msg = get_i18n(default_lang)
    if file is None:
        return [gr.update()] * 9 + [t_msg.get("msg_audio_pls_file", "")]
    try:
        import json
        with open(file.name, 'r') as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) == 9:
                return [float(val) for val in data] + [t_msg.get("msg_audio_loaded", "")]
            else:
                return [gr.update()] * 9 + [t_msg.get("msg_audio_fmt_err", "")]
    except Exception as e:
        return [gr.update()] * 9 + [t_msg.get("msg_audio_load_err", "").format(err=str(e))]

from matplotlib.figure import Figure

def create_analysis_plot(audio_data, fs, title):
    """
    產生與 test_DNN-HA_wavfile.py 一致的分析圖表
    """
    if audio_data is None:
        return None
        
    fig = Figure(figsize=(10, 12))
    axes = fig.subplots(3, 1)
    t = np.arange(len(audio_data)) / fs * 1000 # ms
    
    # 1. 時域波形 (Time Domain)
    axes[0].plot(t, audio_data, linewidth=0.5, color='blue')
    axes[0].set_title(f'Waveform - {title}')
    axes[0].set_xlabel('Time [ms]')
    axes[0].set_ylabel('Sound Pressure [Pa]')
    axes[0].grid(True, linewidth=0.3, linestyle='--')
    
    # 2. 頻譜圖 (Spectrogram)
    freqs, times, spec = sp_sig.spectrogram(audio_data, fs, nperseg=256)
    axes[1].imshow(np.flip(20*np.log10(spec + 1e-10), axis=0), cmap='turbo', 
                   extent=(0, t[-1], freqs[0]/1000, freqs[-1]/1000), aspect="auto")
    axes[1].set_title(f'Spectrogram - {title}')
    axes[1].set_xlabel('Time [ms]')
    axes[1].set_ylabel('Frequency [kHz]')
    
    # 3. 功率譜/倍頻程分析 (Power Spectrum / Octave Band)
    if PyOctaveBand:
        try:
            spl, freq = PyOctaveBand.octavefilter(audio_data, fs=fs, fraction=3, order=6, limits=[10, 8000], show=0)
            axes[2].semilogx(freq, spl, linewidth=1.0, color='red')
            axes[2].set_title(f'1/3 Octave Band Analysis - {title}')
            axes[2].set_xlabel('Frequency [Hz]')
            axes[2].set_ylabel('Magnitude [dB]')
            axes[2].set_xlim([10, 10000])
            axes[2].grid(True, which='both', linewidth=0.3, linestyle='--')
        except Exception as e:
            axes[2].text(0.5, 0.5, f"Octave Analysis Error: {e}", ha='center')
    else:
        f_fft = np.fft.rfftfreq(len(audio_data), 1/fs)
        m_fft = 20 * np.log10(np.abs(np.fft.rfft(audio_data)) + 1e-10)
        axes[2].semilogx(f_fft, m_fft, linewidth=0.5, color='red')
        axes[2].set_title(f'FFT Spectrum - {title}')
        axes[2].set_xlabel('Frequency [Hz]')
        axes[2].set_ylabel('Magnitude [dB]')
        axes[2].set_xlim([10, 10000])
        axes[2].grid(True, which='both', linewidth=0.3, linestyle='--')

    plt.tight_layout()
    return fig

from matplotlib.patches import Polygon
import matplotlib

def plot_audiogram_figure(*losses):
    try:
        losses = [float(l) if l is not None else 0.0 for l in losses]
    except:
        losses = [0]*9
        
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'SimHei', 'Arial']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    import io
    from PIL import Image
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    
    fig = Figure(figsize=(6, 4))
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    freqs = [125, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000]
    
    ax.set_ylim(120, -10)
    ax.set_yticks(np.arange(-10, 121, 10))
    
    x_ticks_pos = {125:0, 250:1, 500:2, 750:2.5, 1000:3, 1500:3.5, 2000:4, 3000:4.5, 4000:5, 6000:5.5, 8000:6}
    x_pos = [x_ticks_pos[f] for f in freqs]
    
    ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
    ax.set_xlim(-0.5, 6.5)
    ax.set_xticks(list(x_ticks_pos.values()))
    ax.set_xticklabels([str(k) for k in x_ticks_pos.keys()])
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Freq (Hz)', loc='right')
    ax.set_ylabel('Loss (dB)')
    ax.set_title("Audiogram", fontsize=16, pad=20)
    
    ax2 = ax.twinx()
    ax2.set_ylim(120, -10)
    ranges = [(-10, 20, 'Normal'), (20, 40, 'Mild'), (40, 55, 'Moderate'), 
              (55, 70, 'Mod-Severe'), (70, 90, 'Severe'), (90, 120, 'Profound')]
    
    yticks2 = []
    ylabels2 = []
    for r in ranges:
        mid = (r[0] + r[1]) / 2
        yticks2.append(mid)
        ylabels2.append(r[2])
    
    ax2.set_yticks(yticks2)
    ax2.set_yticklabels(ylabels2, fontdict={'fontsize': 10})
    ax2.tick_params(axis='y', length=0)
    
    for r in ranges[1:]:
        ax2.axhline(r[0], color='gray', linestyle=':', linewidth=0.5)

    import scipy.interpolate as spi

    xt = np.array([0.7, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.3])
    yt = np.array([36,  22,  30,  34,  30,  24,  18,  24])
    
    xb = np.array([0.7, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.3])
    yb = np.array([36,  50,  58,  60,  56,  45,  30,  24])
    
    try:
        interp_top = spi.PchipInterpolator(xt, yt)
        interp_bot = spi.PchipInterpolator(xb, yb)
        
        x_dense = np.linspace(0.7, 6.3, 100)
        y_dense_top = interp_top(x_dense)
        y_dense_bot = interp_bot(x_dense)
        
        smooth_x = np.concatenate([x_dense, x_dense[::-1]])
        smooth_y = np.concatenate([y_dense_top, y_dense_bot[::-1]])
    except Exception as e:
        smooth_x = np.concatenate([xt, xb[::-1]])
        smooth_y = np.concatenate([yt, yb[::-1]])

    banana_polygon = Polygon(
        xy=list(zip(smooth_x, smooth_y)),
        closed=True, color='lightgray', alpha=0.5, edgecolor=None
    )
    ax.add_patch(banana_polygon)
    
    ax.plot(x_pos, losses, marker='o', markersize=8, markerfacecolor='none', 
            markeredgecolor='red', markeredgewidth=2,
            linestyle='-', color='red', linewidth=2)
            
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    return Image.open(buf)

initial_audiogram_image = plot_audiogram_figure(*default_audiogram)

custom_head = """
<script>
window.toneAudioCtx = null;
window.toneOscillator = null;
window.toneGainNode = null;
window.toneIsPlaying = false;

window.initToneAudio = function() {
    if (!window.toneAudioCtx) {
        window.toneAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
        window.toneGainNode = window.toneAudioCtx.createGain();
        window.toneGainNode.connect(window.toneAudioCtx.destination);
    }
};

window.toggleTone = function(freq, vol, wave) {
    window.initToneAudio();
    if (window.toneAudioCtx.state === 'suspended') {
        window.toneAudioCtx.resume();
    }
    if (window.toneIsPlaying) {
        if (window.toneOscillator) {
            window.toneOscillator.stop();
            window.toneOscillator.disconnect();
        }
        window.toneIsPlaying = false;
        return "PLAY";
    } else {
        window.toneOscillator = window.toneAudioCtx.createOscillator();
        window.toneOscillator.type = wave || 'sine';
        window.toneOscillator.frequency.setValueAtTime(freq, window.toneAudioCtx.currentTime);
        window.toneGainNode.gain.setValueAtTime(vol / 100, window.toneAudioCtx.currentTime);
        window.toneOscillator.connect(window.toneGainNode);
        window.toneOscillator.start();
        window.toneIsPlaying = true;
        return "STOP";
    }
};

window.updateTone = function(freq, vol, wave) {
    if (window.toneOscillator && window.toneIsPlaying) {
        window.toneOscillator.frequency.setTargetAtTime(freq, window.toneAudioCtx.currentTime, 0.05);
        window.toneOscillator.type = wave || 'sine';
        window.toneGainNode.gain.setTargetAtTime(vol / 100, window.toneAudioCtx.currentTime, 0.05);
    }
};
</script>
"""

with gr.Blocks(title="DNN-HA Web UI") as demo:
    t = get_i18n(default_lang)

    state_file_orig = gr.State(None)
    state_file_sim = gr.State(None)
    state_file_proc = gr.State(None)
    state_mic_last = gr.State(None)
    
    ui_title = gr.Markdown(t.get("title"))
    ui_desc = gr.Markdown(t.get("desc"))

    with gr.Tabs() as tabs:
        with gr.Tab(t.get("tab_file"), id="tab_file") as tab_file:
            with gr.Row():
                with gr.Column(scale=1):
                    ui_file_step1 = gr.Markdown(t.get("file_step1"))
                    with gr.Group():
                        in_audio = gr.Audio(type="filepath", label=t.get("in_audio_lbl"), value="wavfiles/00131.wav")
                        btn_ana_orig = gr.Button(t.get("btn_ana_orig"), size="sm")
                    
                    with gr.Group():
                        ui_file_basic_cfg = gr.Markdown(t.get("file_basic_cfg"))
                        in_L = gr.Slider(minimum=30, maximum=110, value=70, step=1, label=t.get("in_L_lbl"))
                        in_sim_gain = gr.Slider(minimum=-40, maximum=40, value=0, step=1, label=t.get("in_sim_gain_lbl"))
                        in_SNR = gr.Textbox(value="", label=t.get("in_SNR_lbl"))
                        in_frame = gr.Number(value=0, label=t.get("in_frame_lbl"))
                        
                    btn_process = gr.Button(t.get("btn_process"), variant="primary")
                    
                with gr.Column(scale=1):
                    ui_file_step2 = gr.Markdown(t.get("file_step2"))
                    out_status = gr.Textbox(label=t.get("out_status_lbl"), lines=4)
                    
                    with gr.Group():
                        out_sim_audio = gr.Audio(label=t.get("out_sim_audio_lbl"), interactive=False)
                        btn_ana_sim = gr.Button(t.get("btn_ana_sim"), size="sm")
                        
                    with gr.Group():
                        out_audio = gr.Audio(label=t.get("out_audio_lbl"), interactive=False)
                        btn_ana_proc = gr.Button(t.get("btn_ana_proc"), size="sm")

        with gr.Tab(t.get("tab_mic"), id="tab_mic") as tab_mic:
            with gr.Row():
                with gr.Column(scale=1):
                    ui_mic_step1 = gr.Markdown(t.get("mic_step1"))
                    mic_input = gr.Audio(sources=["microphone"], streaming=True, label=t.get("mic_input_lbl"))
                    
                    with gr.Group():
                        ui_mic_mode_title = gr.Markdown(t.get("mic_mode_title"))
                        mic_mode = gr.Radio(
                            choices=[t.get("mic_mode_c1"), t.get("mic_mode_c2"), t.get("mic_mode_c3")],
                            value=t.get("mic_mode_c3"),
                            type="index",
                            label=t.get("mic_mode_lbl")
                        )
                    
                    mic_L = gr.Slider(minimum=30, maximum=110, value=70, step=1, label=t.get("mic_L_lbl"))
                    mic_sim_gain = gr.Slider(minimum=-40, maximum=40, value=0, step=1, label=t.get("mic_sim_gain_lbl"))
                    ui_mic_tip = gr.Markdown(t.get("mic_tip"))

                with gr.Column(scale=1):
                    ui_mic_step2 = gr.Markdown(t.get("mic_step2"))
                    mic_output = gr.Audio(streaming=True, label=t.get("mic_output_lbl"), autoplay=True)
                    btn_ana_mic = gr.Button(t.get("btn_ana_mic"), variant="secondary")
                    ui_mic_warn = gr.Markdown(t.get("mic_warn"))

        with gr.Tab(t.get("tab_analysis"), id="tab_analysis") as tab_analysis:
            ui_ana_title = gr.Markdown(t.get("ana_title"))
            ui_ana_desc = gr.Markdown(t.get("ana_desc"))
            analysis_plot = gr.Plot(label=t.get("analysis_plot_lbl"))

        with gr.Tab(t.get("tab_audiogram")) as tab_audiogram:
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        ui_audio_title = gr.Markdown(t.get("audio_title"))
                        audio_inputs = []
                        for i in range(0, 9, 3):
                            with gr.Row():
                                for j in range(3):
                                    idx = i + j
                                    num_input = gr.Number(value=default_audiogram[idx], label=f"{FREQS[idx]} Hz")
                                    audio_inputs.append(num_input)
                        
                        with gr.Row():
                            btn_save_audio = gr.Button(t.get("btn_save_audio"), size="sm")
                            btn_load_audio = gr.UploadButton(t.get("btn_load_audio"), file_types=[".json"], size="sm")
                            btn_update_plot = gr.Button(t.get("btn_update_plot"), size="sm")
                            save_audio_status = gr.Markdown("")
                            
                with gr.Column(scale=1):
                    audiogram_plot = gr.Image(value=initial_audiogram_image, label=t.get("audiogram_plot_lbl"), type="pil", interactive=False)

        with gr.Tab(t.get("tab_tone"), id="tab_tone") as tab_tone:
            ui_tone_title = gr.Markdown(t.get("tone_title"))
            ui_tone_desc = gr.Markdown(t.get("tone_desc"))
            
            with gr.Row():
                with gr.Column(scale=8):
                    tone_freq_slider = gr.Slider(minimum=1, maximum=20000, value=440, step=1, label=t.get("tone_freq_lbl"))
                with gr.Column(scale=1, min_width=100):
                    tone_freq_number = gr.Number(value=440, label=t.get("tone_freq_num_lbl"))
                    
            with gr.Row():
                btn_tone_half = gr.Button(t.get("btn_tone_half"))
                btn_tone_minus = gr.Button(t.get("btn_tone_minus"))
                btn_tone_plus = gr.Button(t.get("btn_tone_plus"))
                btn_tone_double = gr.Button(t.get("btn_tone_double"))
                
            with gr.Row():
                with gr.Column(scale=2):
                    tone_vol_slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label=t.get("tone_vol_lbl"))
                    btn_tone_play = gr.Button(t.get("btn_tone_play"), variant="primary", size="lg")
                with gr.Column(scale=1):
                    tone_waveform = gr.Radio(["sine", "square", "sawtooth", "triangle"], value="sine", label=t.get("tone_waveform_lbl"))

    gr.Markdown("""<hr>Copyright (c) 2026 CHAO-CHIA, LIU. All rights reserved.<br>This interface is developed by CHAO-CHIA, LIU. The core processing logic and DNN models are derived from the <a href="https://github.com/fotisdr/DNN-HA" target="_blank">fotisdr/DNN-HA</a> project.<br><br><small><b>REQUIRED UGent ACADEMIC LICENSE NOTICE:</b><br>© copyright 2020 Ghent University – Universiteit Gent, all rights reserved; this Derivative work is made available for non-commercial academic research purposes and subject to an UGent Academic License.</small>""")

    # --- 事件串接 ---
    
    # 檔案處理模式
    all_file_inputs = [in_audio, in_L, in_SNR, in_frame, in_sim_gain] + audio_inputs
    btn_process.click(
        fn=process_audio,
        inputs=all_file_inputs,
        outputs=[out_audio, out_sim_audio, out_status, state_file_orig, state_file_sim, state_file_proc]
    )
    
    # 檔案分析按鈕
    btn_ana_orig.click(
        fn=lambda data: create_analysis_plot(data, FS_MODEL, "Original Input (File Mode)"),
        inputs=[state_file_orig],
        outputs=[analysis_plot]
    ).then(lambda: gr.update(selected="tab_analysis"), outputs=tabs)

    btn_ana_sim.click(
        fn=lambda data: create_analysis_plot(data, FS_MODEL, "Simulated Hearing Loss (File Mode)"),
        inputs=[state_file_sim],
        outputs=[analysis_plot]
    ).then(lambda: gr.update(selected="tab_analysis"), outputs=tabs)

    btn_ana_proc.click(
        fn=lambda data: create_analysis_plot(data, FS_MODEL, "AI Compensated (File Mode)"),
        inputs=[state_file_proc],
        outputs=[analysis_plot]
    ).then(lambda: gr.update(selected="tab_analysis"), outputs=tabs)

    # 麥克風即時模式
    all_mic_inputs = [mic_input, mic_mode, mic_L, mic_sim_gain] + audio_inputs
    mic_input.stream(
        fn=stream_process,
        inputs=all_mic_inputs,
        outputs=[mic_output, state_mic_last],
        show_progress="hidden"
    )

    # 麥克風分析按鈕
    btn_ana_mic.click(
        fn=lambda data, mode: create_analysis_plot(data, FS_MODEL, f"Real-time Mode {mode}"),
        inputs=[state_mic_last, mic_mode],
        outputs=[analysis_plot]
    ).then(lambda: gr.update(selected="tab_analysis"), outputs=tabs)

    # --- 純音產生器事件串接 ---
    btn_tone_play.click(
        fn=None,
        inputs=[tone_freq_slider, tone_vol_slider, tone_waveform],
        outputs=[btn_tone_play],
        js="(f, v, w) => { return [window.toggleTone(f, v, w)]; }"
    )
    
    tone_freq_slider.change(
        fn=None,
        inputs=[tone_freq_slider, tone_vol_slider, tone_waveform],
        outputs=[tone_freq_number],
        js="(f, v, w) => { window.updateTone(f, v, w); return [f]; }"
    )
    
    tone_freq_number.change(
        fn=None,
        inputs=[tone_freq_number, tone_vol_slider, tone_waveform],
        outputs=[tone_freq_slider],
        js="(f, v, w) => { window.updateTone(f, v, w); return [f]; }"
    )
    
    tone_vol_slider.change(
        fn=None,
        inputs=[tone_freq_slider, tone_vol_slider, tone_waveform],
        js="(f, v, w) => { window.updateTone(f, v, w); }"
    )
    
    tone_waveform.change(
        fn=None,
        inputs=[tone_freq_slider, tone_vol_slider, tone_waveform],
        js="(f, v, w) => { window.updateTone(f, v, w); }"
    )
    
    btn_tone_half.click(fn=lambda x: max(1, x/2), inputs=[tone_freq_slider], outputs=[tone_freq_slider])
    btn_tone_double.click(fn=lambda x: min(20000, x*2), inputs=[tone_freq_slider], outputs=[tone_freq_slider])
    btn_tone_minus.click(fn=lambda x: max(1, x-1), inputs=[tone_freq_slider], outputs=[tone_freq_slider])
    btn_tone_plus.click(fn=lambda x: min(20000, x+1), inputs=[tone_freq_slider], outputs=[tone_freq_slider])
    
    # 聽力圖儲存
    btn_save_audio.click(
        fn=save_audiogram_to_file,
        inputs=audio_inputs,
        outputs=save_audio_status
    )

    # 聽力圖讀取
    btn_load_audio.upload(
        fn=load_audiogram_from_file,
        inputs=[btn_load_audio],
        outputs=audio_inputs + [save_audio_status]
    ).then(
        fn=plot_audiogram_figure,
        inputs=audio_inputs,
        outputs=audiogram_plot
    )

    # 聽力圖更新事件
    btn_update_plot.click(
        fn=plot_audiogram_figure,
        inputs=audio_inputs,
        outputs=audiogram_plot
    )
    for num_input in audio_inputs:
        num_input.submit(
            fn=plot_audiogram_figure,
            inputs=audio_inputs,
            outputs=audiogram_plot
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", inbrowser=True, head=custom_head)
