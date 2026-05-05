# 基於深度神經網路 (DNN) 的即時助聽器處理策略 - 網頁測試平台 (DNN-HA Web GUI)

[English](../README.md) | [繁體中文](README_tw.md)

本專案是一個由 **CHAO-CHIA, LIU** 開發與維護的**衍生測試平台**。其底層的核心音訊處理邏輯與深度神經網路（DNN）模型，是完全基於原作者 [Fotios Drakopoulos 等人所發表的 DNN-HA 專案](https://github.com/fotisdr/DNN-HA) 所改寫與封裝。

原專案的 DNN 策略能夠使用「單一」的模型架構，針對不同使用者的聽力圖（Audiogram）提供個人化的音訊補償。本專案則在此強大的基礎上，開發了具備高度互動性的 Web 圖形化介面（GUI），大幅降低了測試與展示的門檻。

## 💡 本專案的貢獻與特色 (What's New in this Project?)

雖然核心 AI 模型來自原作者，但本衍生專案（由 CHAO-CHIA, LIU 開發）重點加入了以下實用功能，解決了原先純 CLI 測試或舊版桌面應用程式常見的跨平台驅動問題：

1. **全新 Web 圖形化介面 (Gradio)**：無須繁瑣指令，提供直覺的參數調整與視覺化回饋。
2. **麥克風即時串流模式 (Real-time Mode)**：完美發揮原模型極低延遲的特性，支援即時收取麥克風音訊，並可自由切換試聽「原始聲音」、「聽損模擬聲音」與「AI 助聽器補償後的聲音」。
3. **高階數據與圖表分析**：內建時域波形 (Waveform)、頻譜圖 (Spectrogram) 及 1/3 倍頻程分析 (1/3 Octave Band) 等專業圖表，並加入「語言香蕉區 (Speech Banana)」作為臨床參考。
4. **純音產生器 (Pure Tone Generator)**：整合 Web Audio API 的即時音訊合成器，方便快速進行聽力測試或設備校準。
5. **多語系介面 (Multilingual Interface)**：支援**英文 (en_US)** 與**繁體中文 (zh_TW)**。系統會自動偵測作業系統語言，並提供下拉選單進行即時切換。

## 🚀 如何啟動本測試平台

請先參考 [docs/install_pkgs.md](docs/install_pkgs.md) 的指示，建立 Python 虛擬環境並安裝相依套件。

### 啟動 Web 介面 (推薦)

根據您的作業系統執行對應腳本：
- **Linux**: `./rungui.sh`
- **Windows**: `rungui.bat`

執行後，在瀏覽器開啟顯示的本地網址（預設為 `http://127.0.0.1:7860`）。

### 啟動傳統指令列測試

若您仍想使用原專案風格的批次測試：
- **Linux**: `./run.sh`
- **Windows**: `run.bat`

處理好的聲音會儲存在 `wavfiles` 目錄下。

---

## ⚖️ 核心演算法來源與版權聲明 (Attribution & Citation)

**重要聲明：本專案（網頁測試平台）的開發者 CHAO-CHIA, LIU 並非原 DNN-HA 演算法開發團隊（UCL / UGent）的成員。** 本專案僅為原模型之應用層延伸開發。

本專案所使用的預訓練模型（`CNN-HA-12layers`）、音訊正規化邏輯及倍頻程濾波器等核心技術，皆歸功於以下原始研究與作者：

*   **原作者**：Fotios Drakopoulos, Arthur Van Den Broucke, Sarah Verhulst
*   **原始專案 GitHub**：[https://github.com/fotisdr/DNN-HA](https://github.com/fotisdr/DNN-HA)

如果您在學術研究中使用了此系統的核心模型，請務必引用原作者的文獻：

> F. Drakopoulos, A. Van Den Broucke and S. Verhulst, "A DNN-Based Hearing-Aid Strategy For Real-Time Processing: One Size Fits All," ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece, 2023, pp. 1-5, doi: 10.1109/ICASSP49357.2023.10094887.

或引用 Zenodo 版本：
> Fotios Drakopoulos, Arthur Van Den Broucke, & Sarah Verhulst. (2023). DNN-HA: A DNN-based hearing-aid strategy for real-time processing (v1.0). Zenodo. https://doi.org/10.5281/zenodo.7717218

若有關於「核心神經網路演算法」的學術或技術問題，請聯繫原作者團隊。若有關於「Web UI 操作、介面程式碼」的問題，則屬於本專案的範疇。

**REQUIRED UGent ACADEMIC LICENSE NOTICE:**
> © copyright 2020 Ghent University – Universiteit Gent, all rights reserved; this Derivative work is made available for non-commercial academic research purposes and subject to an UGent Academic License (https://github.com/fotisdr/DNN-HA/blob/main/LICENSE.txt)
