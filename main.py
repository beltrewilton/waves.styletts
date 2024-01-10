import sys
import os
p = f"{os.getcwd()}/api"
os.chdir(p)
sys.path.append(f"{os.getcwd()}/../StyleTTS2")
os.environ["PHONEMIZER_ESPEAK_LIBRARY"]="/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib"


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import api.api_main as api_main

app = FastAPI()

base_path = os.getcwd()
key_pem = os.getcwd() + '/../certs/key.pem'
public_pem = os.getcwd() + '/../certs/public.crt'

app.include_router(api_main.router)

origins = [
        "http://localhost:8000",
        "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8060, ssl_keyfile=key_pem, ssl_certfile=public_pem)
