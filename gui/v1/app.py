import os
import sys

# 強制設定 TensorFlow 使用舊版 Keras 引擎
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import simpleaudio as sa
import numpy as np
import scipy.signal as sp_sig
import scipy.io.wavfile as sio_wav
import time

# 延遲匯入 TF 以免影響啟動速度，但在這裡先匯入也無妨
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# 匯入 sys 目錄中的共用函式
from extra_functions import wavfile_read, rms, slice_1dsignal, reconstruct_wav

class DNNHA_App:
    def __init__(self, root):
        self.root = root
        self.root.title("DNN-HA 即時助聽器處理系統")
        self.root.geometry("650x550")
        self.root.resizable(False, False)
        
        # 狀態變數
        self.wavfile_input = tk.StringVar(value="wavfiles/00131.wav")
        self.output_wav = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="就緒")
        
        self.audiogram_vars = []
        default_audiogram = [30, 30, 30, 30, 31.6, 32.58, 33.28, 34.28, 35]
        self.freqs = [125, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000]
        
        self.L_var = tk.DoubleVar(value=70.0)
        self.SNR_var = tk.StringVar(value="")
        self.frame_size_var = tk.IntVar(value=0)
        
        self.play_obj = None
        self.create_widgets(default_audiogram)
        
    def create_widgets(self, default_audiogram):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- 檔案選擇區 ---
        frame_file = ttk.LabelFrame(main_frame, text="音訊檔案", padding="10")
        frame_file.pack(fill=tk.X, pady=5)
        
        ttk.Label(frame_file, text="輸入音檔:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame_file, textvariable=self.wavfile_input, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(frame_file, text="瀏覽...", command=self.select_file).grid(row=0, column=2)
        
        # --- 參數設定區 ---
        frame_params = ttk.LabelFrame(main_frame, text="參數設定", padding="10")
        frame_params.pack(fill=tk.X, pady=5)
        
        ttk.Label(frame_params, text="目標音壓 (L, dB SPL):").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(frame_params, textvariable=self.L_var, width=10).grid(row=0, column=1, sticky="w", pady=2)
        
        ttk.Label(frame_params, text="訊噪比 (SNR):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(frame_params, textvariable=self.SNR_var, width=10).grid(row=1, column=1, sticky="w", pady=2)
        ttk.Label(frame_params, text="(留空表示不加噪音)").grid(row=1, column=2, sticky="w")
        
        ttk.Label(frame_params, text="Frame Size:").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(frame_params, textvariable=self.frame_size_var, width=10).grid(row=2, column=1, sticky="w", pady=2)
        ttk.Label(frame_params, text="(填 0 代表整段處理，否則填入 2 的次方如 256)").grid(row=2, column=2, sticky="w")
        
        # --- 聽力圖 (Audiogram) 設定區 ---
        frame_audio = ttk.LabelFrame(main_frame, text="聽力圖設定 (Audiogram in dB HL)", padding="10")
        frame_audio.pack(fill=tk.X, pady=5)
        
        for i, freq in enumerate(self.freqs):
            ttk.Label(frame_audio, text=f"{freq} Hz:").grid(row=i//3, column=(i%3)*2, sticky="e", padx=(10, 2), pady=2)
            var = tk.DoubleVar(value=default_audiogram[i])
            self.audiogram_vars.append(var)
            ttk.Entry(frame_audio, textvariable=var, width=6).grid(row=i//3, column=(i%3)*2+1, sticky="w", pady=2)

        # --- 操作區 ---
        frame_actions = ttk.Frame(main_frame, padding="10")
        frame_actions.pack(fill=tk.X, pady=10)
        
        self.btn_play_in = ttk.Button(frame_actions, text="▶ 播放原始音檔", command=lambda: self.play_wav(self.wavfile_input.get()))
        self.btn_play_in.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = ttk.Button(frame_actions, text="⏹ 停止播放", command=self.stop_play)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        self.btn_process = ttk.Button(frame_actions, text="⚙️ 開始處理", command=self.start_processing)
        self.btn_process.pack(side=tk.LEFT, padx=20)
        
        self.btn_play_out = ttk.Button(frame_actions, text="▶ 播放處理後音檔", command=lambda: self.play_wav(self.output_wav.get()))
        self.btn_play_out.pack(side=tk.LEFT, padx=5)
        self.btn_play_out.config(state=tk.DISABLED)
        
        # --- 狀態列 ---
        ttk.Label(main_frame, textvariable=self.status_var, foreground="blue").pack(side=tk.BOTTOM, pady=10)
        
    def select_file(self):
        filename = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if filename:
            self.wavfile_input.set(filename)
            self.output_wav.set("")
            self.btn_play_out.config(state=tk.DISABLED)

    def play_wav(self, filepath):
        self.stop_play()
        if not filepath or not os.path.exists(filepath):
            messagebox.showwarning("警告", f"找不到音檔: {filepath}")
            return
            
        try:
            wave_obj = sa.WaveObject.from_wave_file(filepath)
            self.play_obj = wave_obj.play()
        except Exception as e:
            messagebox.showerror("錯誤", f"播放失敗: {e}\n(請確認檔案是否為標準 wav 格式)")

    def stop_play(self):
        if self.play_obj and self.play_obj.is_playing():
            self.play_obj.stop()

    def start_processing(self):
        # 收集參數並防呆
        try:
            audiogram = [var.get() for var in self.audiogram_vars]
            L = self.L_var.get()
            snr_str = self.SNR_var.get().strip()
            snr = float(snr_str) if snr_str else ''
            frame_size = self.frame_size_var.get()
            wav_in = self.wavfile_input.get()
        except ValueError:
            messagebox.showerror("錯誤", "參數輸入格式有誤，請確認所有欄位均為數值！")
            return
            
        if not os.path.exists(wav_in):
            messagebox.showerror("錯誤", "輸入音檔不存在！")
            return
            
        self.btn_process.config(state=tk.DISABLED)
        self.status_var.set("狀態：正在載入模型與處理音訊，請稍候...")
        
        # 在背景執行緒中處理，避免畫面卡死
        threading.Thread(target=self._run_inference, args=(audiogram, L, snr, frame_size, wav_in), daemon=True).start()

    def _run_inference(self, audiogram_input, L, SNR, frame_size, wavfile_input):
        try:
            # 固定參數 (與原腳本相同)
            fs_model = 20e3
            Nenc = 6
            p0 = 2e-5
            overlap = 0
            modeldir = 'CNN-HA-12layers'
            save_wav = 'wavfiles'
            wav_dBref = 110
            
            # --- 音訊前處理 ---
            speechsignal, fs_speech = wavfile_read(wavfile_input) 
            if fs_model != fs_speech :
                stim_full = sp_sig.resample_poly(speechsignal, fs_model, fs_speech)
            else :
                stim_full = speechsignal

            stim_length = stim_full.size
            rem = stim_length % (2**Nenc)
            if rem:
                stim_length = stim_length + int(2**Nenc-rem)

            stim = np.zeros((1,stim_length))
            stim[0,:stim_full.size] = p0 * 10**(L/20) * stim_full / rms(stim_full)

            # --- 加入噪音邏輯 ---
            if SNR != '':
                noise = np.random.normal(size=stim_length)
                speechrms = rms(stim)
                noiserms = rms(noise)
                ratio = (speechrms/noiserms) * np.sqrt(10**(-SNR/10))
                stim[0,:] = stim[0,:] + ratio*noise

            stim = np.expand_dims(stim, axis=2)
            
            # --- 載入模型 ---
            weights_name = "/Gmodel.h5"
            with open(modeldir + "/Gmodel.json", "r") as json_file:
                loaded_model_json = json_file.read()
            
            # 使用 Keras 載入模型
            modelp = model_from_json(loaded_model_json)
            modelp.load_weights(modeldir + weights_name)
            
            # --- 推論處理 ---
            if frame_size:
                stim_cropped = slice_1dsignal(stim[0,:,0], frame_size, 0, stride=1-overlap)
                stim_cropped = np.reshape(stim_cropped, (-1,frame_size,1))
            else:
                stim_cropped = stim
                
            audiogram_rep = np.tile(audiogram_input, (stim_cropped.shape[0], int(stim_cropped.shape[1]/(2**Nenc)), 1))
            
            t = time.time()
            stimp_cropped = modelp.predict([stim_cropped, audiogram_rep])
            proc_time = time.time() - t
            
            if frame_size:
                stimp_cropped = reconstruct_wav(stimp_cropped[:,:,0], stride_factor=1-overlap)
                
            stimp = np.zeros((1, stim.shape[0], stim.shape[1], stim.shape[2]))
            stimp[0,0] = np.reshape(stimp_cropped[:,:stim.shape[1]], (1,-1,1))
            
            # --- 儲存檔案 ---
            if not os.path.exists(save_wav):
                os.makedirs(save_wav)
                
            out_filename = f"{save_wav}/{os.path.basename(wavfile_input)[:-4]}_GUI_processed_20k.wav"
            sio_wav.write(out_filename, int(fs_model), stimp[0,0,:,0] * 10**(-(wav_dBref+20*np.log10(p0))/20))
            
            # --- 更新 UI 狀態 ---
            self.root.after(0, self._process_complete, out_filename, proc_time)
            
        except Exception as e:
            self.root.after(0, self._process_error, str(e))

    def _process_complete(self, out_filename, proc_time):
        self.output_wav.set(out_filename)
        self.status_var.set(f"狀態：處理完成！耗時 {proc_time:.4f} 秒。")
        self.btn_process.config(state=tk.NORMAL)
        self.btn_play_out.config(state=tk.NORMAL)
        messagebox.showinfo("完成", f"處理完畢！\n輸出檔案已儲存至：\n{out_filename}")

    def _process_error(self, err_msg):
        self.status_var.set("狀態：處理發生錯誤！")
        self.btn_process.config(state=tk.NORMAL)
        messagebox.showerror("執行錯誤", f"處理過程中發生錯誤：\n{err_msg}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DNNHA_App(root)
    root.mainloop()
