# 專案 Python 檔案分析文件 (Project Python Files Specification)

本專案主要用於展示與測試基於深度神經網路（DNN）的助聽器（Hearing-Aid, HA）音訊處理策略。專案內包含多個主要的 Python 檔案，分別負責 Web 前端圖形化介面、主程式測試流程、音訊處理輔助函數，以及頻帶分析濾波器。

以下為各個 Python 檔案的詳細功能分析：

## 1. Web GUI 模組 (`gui/` 目錄)
基於 Gradio 框架開發的前端介面，旨在提供跨平台穩定的音訊展示與測試環境。目前已採用 MVC 架構進行模組化重構：

*   **`gui/web_app.py` (主程式 / UI 控制器)**：
    負責定義介面佈局與事件綁定。包含「檔案處理」、「即時麥克風串流」、「視覺化數據分析」、「聽力圖設定」與「純音產生器 (Web Audio API)」等分頁。提供無縫的跨平台音訊播放與使用者互動體驗。
*   **`gui/audio_engine.py` (核心邏輯與推論引擎)**：
    負責載入 TensorFlow 模型並執行音訊訊號處理。實作了基於 FIR 濾波器的即時「聽損模擬 (Hearing Loss Simulation)」，並處理檔案模式 (`process_audio`) 與麥克風模式 (`stream_process`) 的「AI 補償」推論。支援動態調整目標音壓 (SPL)、訊噪比 (SNR) 及額外增益。
*   **`gui/visualizer.py` (視覺化與繪圖模組)**：
    負責產生介面上的各式圖表。包含 `create_analysis_plot` (提供時域波形、頻譜圖與 1/3 倍頻程分析) 以及 `plot_audiogram_figure` (利用高效能的 Pillow 渲染聽力圖，並繪製作為臨床參考的「語言香蕉區」)。
*   **`gui/i18n.py` (多語系模組)**：
    實作動態語言切換機制，讀取 `locale/` 語系檔，並支援啟動時自動偵測系統語系。
*   **`gui/v1/` (歷史備份)**：
    存放重構前的單檔巨石版 `web_app.py` 以及舊版 Tkinter `app.py` 程式碼。

## 2. `sys/test_DNN-HA_wavfile.py`
**角色**：主程式 / 測試腳本
**功能說明**：
這是專案的核心執行腳本，負責整合所有元件並實際執行基於 DNN 的助聽器補償模型。
*   **參數設定**：可設定輸入音檔（如 `00131.wav`）、模擬的聽力圖（Audiogram，定義於 125Hz 到 8000Hz 等 8 個頻率點）、目標音壓位準（SPL）、訊噪比（SNR）以及分幀處理的參數（frame_size, overlap）。
*   **音訊預處理**：讀取音檔後，根據設定將取樣率轉換為模型所需的頻率（20kHz），調整訊號音量至指定的 SPL，並可選擇性地加入環境噪音。
*   **載入模型與推論**：載入位於 `CNN-HA-12layers` 目錄下的 Keras/TensorFlow 預訓練模型（`Gmodel.json` 與 `Gmodel.h5`），並將音訊與聽力圖資料輸入模型以進行即時推論（支援整段音訊一次推論或分幀推論）。
*   **結果輸出與視覺化**：將處理過後的音訊輸出為新的 `.wav` 檔案。並使用 `matplotlib` 繪製多種圖表進行成效對比，包含：處理前後的時域波形圖（Time-domain signal）、頻譜圖（Spectrogram）、輸入的聽力圖（Audiogram input），以及倍頻程頻譜幅度圖（Magnitude spectrum）。

## 3. `sys/extra_functions.py`
**角色**：輔助工具模組
**功能說明**：
包含了一系列處理音訊陣列的工具函數，主要支援主程式進行音訊資料的轉換與運算。
*   `slice_1dsignal`: 將一維的連續音訊訊號切割成多個視窗（frames/windows），可自訂視窗大小與步長比例（stride），以利於進行分段處理。
*   `reconstruct_wav`: 將切割後的音訊矩陣重新組合回一維連續訊號。在重組過程中會考慮疊加部分（overlap-add）並作相應的縮放還原，確保接縫平順。
*   `rms`: 計算音訊訊號的均方根值（Root Mean Square），常用於計算音量大小。
*   `next_power_of_2`: 輔助運算，尋找並回傳大於等於給定數值的下一個 2 的次方數。
*   `wavfile_read`: 封裝了 `scipy.io.wavfile.read`，讀取音檔時會自動將各種格式（如 16-bit, 32-bit int）的數值正規化至 `[-1.0, 1.0]` 的浮點數區區間，並支援在讀取時自動重新取樣至指定的取樣率。

## 4. `sys/PyOctaveBand.py`
**角色**：頻帶濾波分析模組
**功能說明**：
實作了倍頻程（Octave-Band）與分數倍頻程（Fractional Octave-Band）濾波器。這個檔案在主程式中主要用於計算與繪製處理前後的頻譜分佈差異。
*   **核心運算**：使用二階截面（Second-Order Sections, SOS）係數的 Butterworth 濾波器來對訊號進行頻帶過濾，並透過向下取樣（downsampling）來增進濾波器效能。
*   `octavefilter`: 模組的主要 API，用於將訊號通過多個（分數）倍頻帶濾波器，並計算各個頻帶的聲壓級（SPL）和對應的頻率陣列。
*   `getansifrequencies` / `normalizedfreq`: 根據 ANSI S1.11-2004 與 IEC 61260-1-2014 標準，計算並產生標準的倍頻程或三分之一倍頻程的中心頻率、下邊界頻率及上邊界頻率。

## 5. `gui/v1/app.py`
**角色**：舊版桌面圖形化介面 (Tkinter GUI)
**功能說明**：
這是專案早期的基於 Tkinter 的桌面應用程式介面，目前已作為備用或參考。
*   **介面佈局**：提供設定目標 SPL、聽力圖參數、訊噪比等輸入欄位。
*   **本地播放限制**：依賴 `simpleaudio` 進行本機音訊播放。由於跨平台（如 Linux 或特定環境下）容易發生音效驅動不相容（Segmentation fault）等問題，已建議改用基於瀏覽器的 `gui/web_app.py`。
*   **整合測試邏輯**：內部包含了與 `test_DNN-HA_wavfile.py` 相似的處理與推論流程，但介面與功能維護已轉移至 Web 版。

## 6. `locale/`
**角色**：語系資源目錄
**功能說明**：
儲存各國語系的 JSON 檔案，供 `gui/web_app.py` 進行國際化（i18n）渲染。
*   `zh_TW.json`: 繁體中文語系檔。
*   `en_US.json`: 英文語系檔。
