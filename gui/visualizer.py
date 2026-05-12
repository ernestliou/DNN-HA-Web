import numpy as np
import scipy.signal as sp_sig
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import matplotlib
import io
import sys
import os
from PIL import Image

# 自動尋找系統中可用的中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'Noto Sans TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # 修復負號顯示問題

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys_dir = os.path.join(project_root, "sys")
if sys_dir not in sys.path:
    sys.path.append(sys_dir)

from gui.i18n import get_i18n

try:
    import PyOctaveBand
except ImportError:
    PyOctaveBand = None

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

    fig.tight_layout()
    return fig

def plot_audiogram_figure(*losses):
    t_msg = get_i18n()
    
    try:
        losses = [float(l) if l is not None else 0.0 for l in losses]
    except:
        losses = [0]*9
        
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'SimHei', 'Arial']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    
    fig = Figure(figsize=(8, 5))
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
    ax.set_title(t_msg.get("plot_title", "Audiogram"), fontsize=16, pad=20)
    
    ax2 = ax.twinx()
    ax2.set_ylim(120, -10)
    
    ranges = [
        (-10, 20, t_msg.get("plot_y_normal", 'Normal')), 
        (20, 40, t_msg.get("plot_y_mild", 'Mild')), 
        (40, 55, t_msg.get("plot_y_moderate", 'Moderate')), 
        (55, 70, t_msg.get("plot_y_mod_severe", 'Mod-Severe')), 
        (70, 90, t_msg.get("plot_y_severe", 'Severe')), 
        (90, 120, t_msg.get("plot_y_profound", 'Profound'))
    ]
    
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
    
    # 補償極限線與區域標示
    ax.axhline(40, color='blue', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.text(6.4, 38, t_msg.get("plot_ai_core", 'AI 核心處理範圍'), color='blue', ha='right', va='bottom', fontsize=10, fontweight='bold')
    
    ax.text(6.4, 70, t_msg.get("plot_dsp_linear", 'DSP 線性補償區'), color='darkorange', ha='right', va='center', fontsize=10, fontweight='bold', alpha=0.8)
    
    # 標示死區 (Dead Regions)
    dead_region_text = t_msg.get("plot_dead_region", '死區')
    for x, l in zip(x_pos, losses):
        if l >= 90:
            ax.fill_between([x-0.15, x+0.15], [l, l], [120, 120], color='red', alpha=0.3)
            ax.text(x, l + 5, dead_region_text, color='darkred', ha='center', va='top', fontsize=9, fontweight='bold')
    
    ax.plot(x_pos, losses, marker='o', markersize=8, markerfacecolor='none', 
            markeredgecolor='red', markeredgewidth=2,
            linestyle='-', color='red', linewidth=2)
            
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    return Image.open(buf)
