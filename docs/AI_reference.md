# AI Reference & Project Context (AI 讀取參考與開發規範)

此文件旨在為 AI 輔助開發工具（如 Antigravity, Cursor, GitHub Copilot）提供專案背景資訊，並作為跨平台（Windows/Linux）遷移時的參考指南。

## 1. 專案基本資訊 (Project Overview)
- **專案名稱**: DNN-HA (DNN-based hearing-aid strategy for real-time processing)
- **核心目標**: 基於深度神經網路的即時助聽器處理策略，支援個人化聽力圖補償。
- **技術棧**: Python 3.9+, TensorFlow/Keras, Gradio (Web UI), Web Audio API (Synthesizer), i18n (JSON-based).
- **關鍵參數**:
  - **模型取樣率 (FS_MODEL)**: 20,000 Hz (20kHz).
  - **模型基本幀長 (Frame size)**: 64 samples (或其倍數).
  - **輸入頻率點**: 125, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000 Hz.

## 2. 目錄結構 (Directory Structure)
```text
DNN-HA/
├── CNN-HA-12layers/     # 預訓練核心模型 (Gmodel.json, Gmodel.h5)
├── gui/                 # 圖形化介面相關 (MVC 架構)
│   ├── web_app.py       # 主程式：Gradio UI 與事件綁定 (推薦入口)
│   ├── audio_engine.py  # 核心邏輯：模型載入與音訊推論處理
│   ├── visualizer.py    # 繪圖模組：分析圖表與聽力圖渲染
│   ├── i18n.py          # 多語系模組：語系解析與載入
│   ├── app.py           # 舊版 Tkinter 介面
│   ├── audiogram.json   # 預設聽力圖儲存檔
│   └── v1/              # 重構前的 web_app.py 歷史備份
├── sys/                 # 核心演算法與系統組件
│   ├── test_DNN-HA_wavfile.py  # 核心 CLI 測試腳本
│   ├── extra_functions.py      # 音訊處理輔助函式
│   └── PyOctaveBand.py         # 1/3 倍頻程分析模組
├── locale/              # 多語系資源檔 (zh_TW.json, en_US.json)
├── docs/                # 專案文件 (Markdown)
│   └── papers/          # 參考文獻目錄
├── wavfiles/            # 預設測試音檔與處理後結果儲存區
├── audiogram/           # 聽力圖 JSON 存檔目錄
├── requirements.txt     # Python 相依套件清單
├── rungui.bat / .sh     # 啟動 Web UI (Win/Linux)
└── run.bat / .sh        # 啟動 CLI 測試 (Win/Linux)
```

## 3. 環境與相依性 (Environment & Dependencies)
- **虛擬環境**: 建議使用 `.venv` 目錄。
- **主要依賴**:
  - `tensorflow`, `tf-keras` (模型執行)
  - `gradio` (Web 介面)
  - `numpy`, `scipy` (訊號處理)
  - `matplotlib`, `Pillow` (數據視覺化)
  - `simpleaudio` (本機播放)

## 4. 關鍵入口點 (Key Entry Points)
### Web 圖形介面 (跨平台穩定)
- 入口: `gui/web_app.py`
- 啟動方式: `python gui/web_app.py` 或執行 `rungui.bat` / `rungui.sh`
- 功能特色: 支援麥克風即時串流、純音產生器、數據分析圖表、多語系切換。

### CLI 批次測試
- 入口: `sys/test_DNN-HA_wavfile.py`
- 啟動方式: `python sys/test_DNN-HA_wavfile.py` 或執行 `run.bat` / `run.sh`

## 5. 開發與遷移規範 (Development & Migration Notes)
- **保留原專案檔案**: `docs/Original_README.md` 是原來的專案說明，請勿更動。
- **環境變數**: 專案強制設定 `os.environ['TF_USE_LEGACY_KERAS'] = '1'` 以相容舊版模型。
- **跨平台注意事項**:
  - 路徑處理應使用 `os.path.join` 以相容 Windows (`\`) 與 Linux (`/`)。
  - 音訊驅動：Linux 下若 `simpleaudio` 報錯，請改用 Web UI 的瀏覽器播放功能。
- **數據分析**: 所有的分析圖表（時域、頻譜、倍頻程）應保持與 `test_DNN-HA_wavfile.py` 的計算邏輯一致。
- **更新 docs**: 任何功能變動後，應同步更新 `docs/spec.md` 與 `docs/tasks.txt`。

## 6. AI 協助開發建議 (Instructions for AI)
- **修改代碼時**: 請保留所有音訊正規化邏輯（P0 = 2e-5, dB SPL 轉換）。
- **新增功能時**: 優先在 Web GUI 中實作以保持 UI 統一。請遵循目前的 MVC 架構（UI 放 `web_app.py`，音訊邏輯放 `audio_engine.py`，圖表放 `visualizer.py`）。
- **排錯時**: 優先檢查取樣率是否轉換為 20kHz，以及模型輸入維度是否正確。
