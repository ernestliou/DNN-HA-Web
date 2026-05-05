# Project Dependency Installation Guide

This project is developed in Python and relies on the following third-party packages to process audio, execute deep learning model inference, and render charts.

## Required Packages List

Based on the code analysis of the project, executing this project requires installing the following external packages:

1. **`tensorflow`** and **`tf-keras`**
   - **Purpose**: Used for loading and executing pre-trained DNN/CNN hearing aid compensation models (Keras `.json` and `.h5` formats). Because the project models were saved by an older version of Keras, `tf-keras` is required to support loading in the newer versions of TensorFlow.
2. **`numpy`**
   - **Purpose**: Provides high-performance multi-dimensional array and matrix operations, used for computing audio data (such as root mean square, array manipulations, etc.).
3. **`scipy`**
   - **Purpose**: Primarily used for reading and writing `.wav` audio files (`scipy.io.wavfile`) and executing advanced audio signal processing, such as resampling and spectrogram analysis (`scipy.signal`).
4. **`matplotlib`**
   - **Purpose**: Used for visualizing the processed results, plotting the audio waveforms, spectrograms, and octave band analysis charts before and after processing.
5. **`gradio`**
   - **Purpose**: Builds a stable and robust Web graphical user interface, responsible for audio playback and parameter adjustment.
6. **`Pillow`**
   - **Purpose**: Used for high-performance Audiogram visualization rendering in the Web GUI.

*(Note: `time`, `os`, and `sys` are built-in Python standard libraries and do not require additional installation.)*

## Installation and Execution (Virtual Environment Recommended)

It is highly recommended to use a Python virtual environment (such as `.venv`) in the project to install these packages. This can prevent version conflicts with system packages or other projects.

### Step 1: Create a Virtual Environment
Open a terminal in the project root directory and execute the following command to create a virtual environment named `.venv`:

```bash
python3 -m venv .venv
```

### Step 2: Activate the Virtual Environment
Depending on your operating system, execute the following command to activate the virtual environment:

*   **Linux / macOS**:
    ```bash
    source .venv/bin/activate
    ```
*   **Windows**:
    ```cmd
    .venv\Scripts\activate
    ```
*(After successful activation, your command line prompt should start with `(.venv)`)*

### Step 3: Install Dependencies
While ensuring the virtual environment is activated, use `pip` to install all required packages listed in `requirements.txt` located in the project root directory:

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
After installation is complete, it is recommended to run the Web-based graphical interface, which provides a more intuitive operation and audio comparison function:

*   **Linux**: `./rungui.sh`
*   **Windows**: `rungui.bat`

*(After starting, please open the local URL displayed in the terminal in your browser)*

If you want to run the traditional terminal test script, please execute:

*   **Linux**: `./run.sh`
*   **Windows**: `run.bat`

### Deactivate the Virtual Environment
When you have finished working on the project and want to exit the virtual environment, simply enter the following in the terminal:

```bash
deactivate
```
