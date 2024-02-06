# import sys
# import os
# sys.path.append(f"/Users/beltre.wilton/apps/waves.styletts/")
# sys.path.append(f"/Users/beltre.wilton/apps/waves.styletts/StyleTTS2")
# os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib"

from pathlib import Path

import torch

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from api.model_loader import inference, LFinference, compute_style, save_wav, stretch_with_rubberband


class Info(BaseModel):
    audio_path: str
    text: str
    alpha: float = 0.3
    beta: float = 0.2
    diffusion_steps: int = 10
    embedding_scale: int = 1

router = APIRouter(prefix='/tts', tags=['tts'])

device = 'cuda' if torch.cuda.is_available() else 'cpu' # MPS causa audio feos

# audio_path = "/Users/beltre.wilton/apps/waves.api/audios"
# audio_path = "/Users/beltre.wilton/apps/waves.api/audios/key"


@router.post("/synth",)
async def synth(info: Info):
    try:
        audio_path = info.audio_path
        noise = torch.randn(1, 1, 256).to(device) # for LInference
        ref_s = compute_style(audio_path)
        alpha = info.alpha
        beta = info.beta
        diffusion_steps = info.diffusion_steps
        embedding_scale = info.embedding_scale
        wav = inference(info.text, ref_s, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
        synth_wav_file = f"{audio_path[:-4]}_synth_a-{alpha}_b-{beta}_df-{diffusion_steps}_em-{embedding_scale}.wav"
        save_wav(synth_wav_file, wav, 24_000)
        stretch_with_rubberband(synth_wav_file, audio_path)

        return {
            "synth_wav_file": synth_wav_file,
            "synth_name": Path(synth_wav_file).name,
            "response": "completed"
        }
    except Exception as ex:
        print(ex)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ex))


def standalone():
    text = "The function returns the number of bytes used in memory by the given Python Tuple object. This is done with the use of get_sizeof that determines how much data can be stored on the device or server memory using the Python built-in module for data."
    try:
        # text = "The check in this very curious act, I got a headache In all the territory of Greenland there are only four traffic lights In the whole island, a man offered us even a few fock guards As a souvenir, they also have fock guards, bear guards A very peculiar light trade"
        abs_path = f"/Users/beltre.wilton/apps/linkedingenai/llama/results/elonmusk.wav"
        noise = torch.randn(1, 1, 256).to(device)
        ref_s = compute_style(abs_path)
        # wav = inference(info.text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1)
        alpha = 0.3
        beta = 0.2
        diffusion_steps = 10
        embedding_scale = 1
        wav = inference(text, ref_s, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps,
                        embedding_scale=embedding_scale)
        synth_wav_file = f"/Users/beltre.wilton/apps/linkedingenai/llama/results/elonmusk_synth.wav"
        save_wav(synth_wav_file, wav, 24_000)

    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    standalone()