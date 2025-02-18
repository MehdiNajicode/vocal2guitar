import io
import os
os.system("chmod 777 ffmpeg")
import torch
import gradio as gr
import librosa
import numpy as np
import soundfile
import logging
from fairseq import checkpoint_utils
from my_utils import load_audio
from vc_infer_pipeline import VC
import traceback
from config import Config
from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from i18n import I18nAuto

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

i18n = I18nAuto()
i18n.print()

config = Config()

models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
    ["hubert_base.pt"],
    suffix="",
)
hubert_model = models[0]
hubert_model = hubert_model.to(config.device)
hubert_model = hubert_model.float()
hubert_model.eval()

global n_spk, tgt_sr, net_g, vc, cpt, version
person = "weights/simple-guitar-crepe-guolv_e1000.pth"
print("loading %s" % person)
cpt = torch.load(person, map_location="cpu")
tgt_sr = cpt["config"][-1]
cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False)
del net_g.enc_q
print(net_g.load_state_dict(cpt["weight"], strict=False))
net_g.eval().to(config.device)
net_g = net_g.float()
vc = VC(tgt_sr, config)
n_spk = cpt["config"][-3]
version="v2"

default_audio=load_audio("logs/mute/1_16k_wavs/mute.wav",16000)
def vc_single(
    input_audio_path,
    f0_up_key,
    f0_method,
    file_index="logs/added_IVF2225_Flat_nprobe_1_simple-guitar-crepe-guolv_v2.index",
    index_rate=1,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=1,
    protect=0.5,
):
    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio_path is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        audio = input_audio_path[1] / 32768.0
        if len(audio.shape) == 2:
            audio = np.mean(audio, -1)
        audio = librosa.resample(audio, orig_sr=input_audio_path[0], target_sr=16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        audio_opt = vc.pipeline(
            model=hubert_model,
            net_g=net_g,
            sid=0,
            audio=audio,
            input_audio_path="123",
            times=times,
            f0_up_key=f0_up_key,
            f0_method=f0_method,
            file_index=file_index,
            index_rate=index_rate,
            if_f0=1,
            filter_radius=filter_radius,
            tgt_sr=tgt_sr,
            resample_sr=resample_sr,
            rms_mix_rate=rms_mix_rate,
            version="v2",
            protect=protect,
            f0_file=None,
        )
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            tgt_sr = resample_sr
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            index_info,
            times[0],
            times[1],
            times[2],
        ), (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return "Error! Details:\n%s" % info, (16000, default_audio)

app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("Simple Vocal to Guitar Demo"):
            gr.Markdown("""
                Adjust pitch shift: Higher values make the guitar sound sharper, lower values make it deeper.
                """)
            vc_input = gr.Audio(label="Upload Audio")
            with gr.Column():
                with gr.Row():
                    vc_transform = gr.Slider(
                        minimum=-12,
                        maximum=12,
                        label="Pitch Shift (Semitones, +12 for higher, -12 for lower)",
                        value=0,
                        step=1,
                        interactive=True,
                    )
                    f0method = gr.Radio(
                        label="Select Pitch Extraction Algorithm (dio for voice, pm for singing)",
                        choices=["pm", "dio"],
                        value="dio",
                        interactive=True,
                    )
                with gr.Row():
                    but = gr.Button("Convert", variant="primary")
                    vc_output1 = gr.Textbox(label="Output Info")
                    vc_output2 = gr.Audio(label="Output Audio (Click three dots for download)")
            but.click(
                vc_single,
                [vc_input, vc_transform, f0method],
                [vc_output1, vc_output2],
            )

app.launch(share=True)  # Enable public link
