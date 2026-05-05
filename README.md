# DNN-Based Hearing-Aid Strategy for Real-Time Processing - Web Testing Platform (DNN-HA Web GUI)

[English](docs/README_en.md) | [繁體中文](docs/README_tw.md)

This project is a **derivative testing platform** developed and maintained by **CHAO-CHIA, LIU**. Its underlying core audio processing logic and deep neural network (DNN) models are entirely based on, adapted from, and wrapped around the [DNN-HA project published by Fotios Drakopoulos et al.](https://github.com/fotisdr/DNN-HA).

The original project's DNN strategy can use a "single" model architecture to provide individualized audio compensation for different users' audiograms. Building upon this powerful foundation, this project introduces a highly interactive Web Graphical User Interface (GUI), significantly lowering the barrier for testing and demonstration.

## 💡 What's New in this Project?

While the core AI model comes from the original authors, this derivative project (developed by CHAO-CHIA, LIU) introduces the following practical features, resolving cross-platform driver issues commonly found in pure CLI testing or older desktop applications:

1. **Brand New Web Graphical Interface (Gradio)**: No need for complex commands. It provides intuitive parameter adjustments and visual feedback.
2. **Real-time Microphone Streaming Mode**: Fully leverages the ultra-low latency characteristics of the original model. It supports real-time microphone audio input and allows you to freely switch and listen to the "Original Sound", "Simulated Hearing Loss Sound", and "AI Compensated Sound".
3. **Advanced Data and Chart Analysis**: Built-in professional charts such as Time-domain Waveform, Spectrogram, and 1/3 Octave Band analysis. It also incorporates the "Speech Banana" as a clinical reference.
4. **Pure Tone Generator**: Integrates a real-time audio synthesizer using the Web Audio API, making it convenient for quick hearing tests or equipment calibration.
5. **Multilingual Interface**: Supports both **English (en_US)** and **Traditional Chinese (zh_TW)**. The system automatically detects the OS language and provides a dropdown menu for seamless real-time switching.

## 🚀 How to Launch this Testing Platform

Please refer to the instructions in [docs/install_pkgs.md](docs/install_pkgs.md) to set up a Python virtual environment and install dependencies.

### Launch the Web Interface (Recommended)

Run the corresponding script based on your operating system:
- **Linux**: `./rungui.sh`
- **Windows**: `rungui.bat`

After running, open the displayed local URL in your browser (default is `http://127.0.0.1:7860`).

### Launch the Traditional Command Line Test

If you still want to use the batch testing in the style of the original project:
- **Linux**: `./run.sh`
- **Windows**: `run.bat`

The processed audio files will be saved in the `wavfiles` directory.

---

## ⚖️ Attribution & Citation

**IMPORTANT DISCLAIMER: The developer of this project (Web testing platform), CHAO-CHIA, LIU, is not a member of the original DNN-HA algorithm development team (UCL / UGent).** This project is merely an application-layer extension developed on top of the original model.

The pre-trained model (`CNN-HA-12layers`), audio normalization logic, and octave-band filters used in this project are credited to the following original research and authors:

*   **Original Authors**: Fotios Drakopoulos, Arthur Van Den Broucke, Sarah Verhulst
*   **Original GitHub Repository**: [https://github.com/fotisdr/DNN-HA](https://github.com/fotisdr/DNN-HA)

If you use the core model of this system in academic research, you must cite the authors' publication:

> F. Drakopoulos, A. Van Den Broucke and S. Verhulst, "A DNN-Based Hearing-Aid Strategy For Real-Time Processing: One Size Fits All," ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece, 2023, pp. 1-5, doi: 10.1109/ICASSP49357.2023.10094887.

Or cite the Zenodo release:
> Fotios Drakopoulos, Arthur Van Den Broucke, & Sarah Verhulst. (2023). DNN-HA: A DNN-based hearing-aid strategy for real-time processing (v1.0). Zenodo. https://doi.org/10.5281/zenodo.7717218

For academic or technical questions regarding the "core neural network algorithm", please contact the original author team. For questions regarding the "Web UI operation and interface code", please refer to this project's scope.

**REQUIRED UGent ACADEMIC LICENSE NOTICE:**
> © copyright 2020 Ghent University – Universiteit Gent, all rights reserved; this Derivative work is made available for non-commercial academic research purposes and subject to an UGent Academic License (https://github.com/fotisdr/DNN-HA/blob/main/LICENSE.txt)
