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
import json
import gradio as gr

# Ensure the root project directory is in sys.path so we can import from 'gui.*' and 'sys.*'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from gui.i18n import get_i18n
from gui.visualizer import create_analysis_plot, plot_audiogram_figure
from gui.audio_engine import process_audio, stream_process, FS_MODEL, FREQS

# --- Gradio UI State & Config ---
AUDIOGRAM_FILE = os.path.join("gui", "audiogram.json")
default_audiogram = [10, 15, 20, 25, 30, 40, 45, 50, 55]

if os.path.exists(AUDIOGRAM_FILE):
    try:
        with open(AUDIOGRAM_FILE, "r") as f:
            saved_audiogram = json.load(f)
            if isinstance(saved_audiogram, list) and len(saved_audiogram) == 9:
                default_audiogram = saved_audiogram
    except Exception as e:
        print(f"Unable to load {AUDIOGRAM_FILE}: {e}")

def save_audiogram_to_file(*args):
    t_msg = get_i18n()
    try:
        with open(AUDIOGRAM_FILE, "w") as f:
            json.dump(list(args), f)
        return t_msg.get("msg_audio_saved", "")
    except Exception as e:
        return t_msg.get("msg_audio_save_err", "").format(err=str(e))

def load_audiogram_from_file(file, is_profound):
    t_msg = get_i18n()
    if file is None:
        return [gr.update()] * 9 + [gr.update(), t_msg.get("msg_audio_pls_file", "")]
    try:
        with open(file.name, 'r') as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) == 9:
                vals = [float(val) for val in data]
                needs_profound = any(v > 70 for v in vals)
                new_profound = True if needs_profound else is_profound
                max_val = 120 if new_profound else 70
                updates = [gr.update(value=min(v, max_val), maximum=max_val) for v in vals]
                return updates + [gr.update(value=new_profound), t_msg.get("msg_audio_loaded", "")]
            else:
                return [gr.update()] * 9 + [gr.update(), t_msg.get("msg_audio_fmt_err", "")]
    except Exception as e:
        return [gr.update()] * 9 + [gr.update(), t_msg.get("msg_audio_load_err", "").format(err=str(e))]

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
    t = get_i18n()

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
                with gr.Column(scale=4):
                    with gr.Group():
                        ui_audio_title = gr.Markdown(t.get("audio_title"))
                        is_profound_default = any(v > 70 for v in default_audiogram)
                        cb_profound = gr.Checkbox(label="啟動極重度聽損模式（Profound Mode）", value=is_profound_default)
                        
                        audio_inputs = []
                        for i in range(0, 9, 2):
                            with gr.Row():
                                for j in range(2):
                                    if i + j < 9:
                                        idx = i + j
                                        max_val = 120 if is_profound_default else 70
                                        num_input = gr.Slider(minimum=-10, maximum=max_val, step=1, value=default_audiogram[idx], label=f"{FREQS[idx]} Hz")
                                        audio_inputs.append(num_input)
                        
                        with gr.Row():
                            btn_save_audio = gr.Button(t.get("btn_save_audio"), size="sm")
                            btn_load_audio = gr.UploadButton(t.get("btn_load_audio"), file_types=[".json"], size="sm")
                            btn_update_plot = gr.Button(t.get("btn_update_plot"), size="sm")
                            save_audio_status = gr.Markdown("")
                            
                with gr.Column(scale=5):
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
    all_file_inputs = [in_audio, in_L, in_SNR, in_frame, in_sim_gain, cb_profound] + audio_inputs
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
    all_mic_inputs = [mic_input, mic_mode, mic_L, mic_sim_gain, cb_profound] + audio_inputs
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
        inputs=[btn_load_audio, cb_profound],
        outputs=audio_inputs + [cb_profound, save_audio_status]
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
        num_input.release(
            fn=plot_audiogram_figure,
            inputs=audio_inputs,
            outputs=audiogram_plot
        )
        
    # 極重度模式切換事件
    def toggle_profound_mode(is_profound, *current_vals):
        max_val = 120 if is_profound else 70
        updates = []
        for v in current_vals:
            new_v = min(v, max_val)
            updates.append(gr.update(maximum=max_val, value=new_v))
        return updates

    cb_profound.change(
        fn=toggle_profound_mode,
        inputs=[cb_profound] + audio_inputs,
        outputs=audio_inputs
    ).then(
        fn=plot_audiogram_figure,
        inputs=audio_inputs,
        outputs=audiogram_plot
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", inbrowser=True, head=custom_head)
