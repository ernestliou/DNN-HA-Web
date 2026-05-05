# Project Python Files Specification 

This project is primarily used to demonstrate and test the deep neural network (DNN) based hearing-aid (HA) audio processing strategy. The project contains multiple main Python files, which are responsible for the Web frontend graphical interface, the main testing workflow, audio processing utility functions, and frequency band analysis filters.

The following is a detailed functional analysis of each Python file:

## 1. Web GUI Modules (`gui/` directory)
The frontend interface developed based on the Gradio framework aims to provide a stable cross-platform audio demonstration and testing environment. It has been recently refactored into a modular MVC architecture:

*   **`gui/web_app.py` (Main Program / UI Controller)**:
    Responsible for defining the interface layout and event binding. It includes tabs for "File Processing", "Real-time Microphone Streaming", "Visual Data Analysis", "Audiogram Settings", and "Pure Tone Generator (Web Audio API)". It provides seamless cross-platform audio playback and user interaction experience.
*   **`gui/audio_engine.py` (Core Logic & Inference Engine)**:
    Responsible for loading the TensorFlow model and executing audio signal processing. It implements real-time "Hearing Loss Simulation" based on FIR filters and handles "AI Compensation" inference for both file mode (`process_audio`) and microphone mode (`stream_process`). It supports dynamic adjustment of target sound pressure level (SPL), signal-to-noise ratio (SNR), and simulation gain.
*   **`gui/visualizer.py` (Visualization & Plotting Module)**:
    Responsible for generating various charts on the interface. It includes `create_analysis_plot` (provides time-domain waveform, spectrogram, and 1/3 octave band analysis) and `plot_audiogram_figure` (renders audiograms using high-performance Pillow, and plots the "Speech Banana" as a clinical reference).
*   **`gui/i18n.py` (Internationalization Module)**:
    Implements a dynamic language switching mechanism, reads locale files from the `locale/` directory, and supports automatic detection of the system language upon startup.
*   **`gui/v1/` (Historical Backup)**:
    Stores the monolithic `web_app.py` and the legacy Tkinter `app.py` code from before the refactoring.

## 2. `sys/test_DNN-HA_wavfile.py`
**Role**: Main Program / Test Script
**Description**:
This is the core execution script of the project, responsible for integrating all components and actually running the DNN-based hearing aid compensation model.
*   **Parameter Settings**: Allows setting the input audio file (e.g., `00131.wav`), the simulated audiogram (defined at 8 frequency points from 125Hz to 8000Hz), the target Sound Pressure Level (SPL), Signal-to-Noise Ratio (SNR), and framing parameters (frame_size, overlap).
*   **Audio Preprocessing**: After reading the audio file, it converts the sampling rate to the frequency required by the model (20kHz) according to the settings, adjusts the signal volume to the specified SPL, and optionally adds environmental noise.
*   **Model Loading and Inference**: Loads the pre-trained Keras/TensorFlow models (`Gmodel.json` and `Gmodel.h5`) located in the `CNN-HA-12layers` directory, and feeds the audio and audiogram data into the model for real-time inference (supports both full-length inference and frame-by-frame inference).
*   **Output and Visualization**: Outputs the processed audio as a new `.wav` file. It also uses `matplotlib` to plot various charts for performance comparison, including: Time-domain signal before and after processing, Spectrogram, Audiogram input, and Magnitude spectrum for octave bands.

## 3. `sys/extra_functions.py`
**Role**: Utility Module
**Description**:
Contains a series of utility functions for processing audio arrays, mainly supporting the main program in audio data conversion and computation.
*   `slice_1dsignal`: Slices a 1D continuous audio signal into multiple frames/windows, allowing customized window size and stride ratio for segmented processing.
*   `reconstruct_wav`: Reconstructs the sliced audio matrix back into a 1D continuous signal. The reconstruction process considers the overlap-add method and performs corresponding scaling to ensure smooth transitions.
*   `rms`: Calculates the Root Mean Square of an audio signal, commonly used for volume calculation.
*   `next_power_of_2`: A helper function that finds and returns the next power of 2 greater than or equal to a given value.
*   `wavfile_read`: Encapsulates `scipy.io.wavfile.read`, automatically normalizing the values of various formats (e.g., 16-bit, 32-bit int) to the `[-1.0, 1.0]` floating-point range when reading audio files, and supports automatic resampling to a specified sampling rate during reading.

## 4. `sys/PyOctaveBand.py`
**Role**: Frequency Band Filtering Analysis Module
**Description**:
Implements Octave-Band and Fractional Octave-Band filters. This file is primarily used in the main program to calculate and plot the difference in spectral distribution before and after processing.
*   **Core Computation**: Uses Butterworth filters with Second-Order Sections (SOS) coefficients to filter the signal by frequency bands, and improves filter performance through downsampling.
*   `octavefilter`: The main API of the module, used to pass the signal through multiple (fractional) octave-band filters and calculate the Sound Pressure Level (SPL) and corresponding frequency array for each band.
*   `getansifrequencies` / `normalizedfreq`: Calculates and generates the standard center frequencies, lower bound frequencies, and upper bound frequencies for octave or one-third octave bands according to ANSI S1.11-2004 and IEC 61260-1-2014 standards.

## 5. `gui/v1/app.py`
**Role**: Legacy Desktop Graphical User Interface (Tkinter GUI)
**Description**:
This is the early Tkinter-based desktop application interface of the project, currently serving as a backup or reference.
*   **Interface Layout**: Provides input fields for setting target SPL, audiogram parameters, SNR, etc.
*   **Local Playback Limitations**: Relies on `simpleaudio` for local audio playback. Due to common audio driver incompatibility issues (such as Segmentation faults) across platforms (like Linux or specific environments), it is recommended to use the browser-based `gui/web_app.py` instead.
*   **Integrated Testing Logic**: Internally contains processing and inference workflows similar to `test_DNN-HA_wavfile.py`, but its interface and feature maintenance have been shifted to the Web version.

## 6. `locale/`
**Role**: Localization Resource Directory
**Description**:
Stores JSON files for various languages, used by `gui/web_app.py` for internationalization (i18n) rendering.
*   `zh_TW.json`: Traditional Chinese locale file.
*   `en_US.json`: English locale file.
