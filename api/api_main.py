import torch

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from api.model_loader import inference, compute_style, save_wav


class Info(BaseModel):
    part: str
    text: str
    alpha: float = 0.3
    beta: float = 0.2
    diffusion_steps: int = 10
    embedding_scale: int = 1

router = APIRouter(prefix='/tts', tags=['tts'])

device = 'cuda' if torch.cuda.is_available() else 'cpu' # MPS causa audio feos

audio_path = "/Users/beltre.wilton/apps/waves.api/audios"


@router.post("/synth",)
async def synth(info: Info):
    try:
        # text = "The check in this very curious act, I got a headache In all the territory of Greenland there are only four traffic lights In the whole island, a man offered us even a few fock guards As a souvenir, they also have fock guards, bear guards A very peculiar light trade"
        abs_path = f"{audio_path}/{info.part}"
        noise = torch.randn(1, 1, 256).to(device)
        ref_s = compute_style(abs_path)
        # wav = inference(info.text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1)
        alpha = info.alpha
        beta = info.beta
        diffusion_steps = info.diffusion_steps
        embedding_scale = info.embedding_scale
        wav = inference(info.text, ref_s, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
        synth_wav_file = f"{abs_path[:-4]}_synth_a-{alpha}_b-{beta}_df-{diffusion_steps}_em-{embedding_scale}.wav"
        synth_name = f"{info.part[:-4]}_synth_a-{alpha}_b-{beta}_df-{diffusion_steps}_em-{embedding_scale}.wav"
        save_wav(synth_wav_file, wav, 24_000)
        return {
            "synth_wav_file": synth_wav_file,
            "synth_name": synth_name,
            "response": "completed"
        }
    except Exception as ex:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ex))
