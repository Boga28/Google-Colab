import os
import time
import re
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

def load_model():
    print("Loading model...")
    config = XttsConfig()   
    config.load_json("/content/xtts2_model/config.json")
    model = Xtts.init_from_config(config)
    config.model_args.kv_cache = True         # Ensure key-value caching is enabled.
    config.model_args.gpt_batch_size = 1        # Adjust batch size as your GPU memory permits.
    model.load_checkpoint(config, checkpoint_dir="/content/xtts2_model/", use_deepspeed=True)
    model.cuda()
    return model

def split_text(text, max_chars=250):
    sentences = sent_tokenize(text)
    chunks = []
    for sent in sentences:
        if len(sent) <= max_chars:
            chunks.append(sent)
        else:
            words = sent.split()
            current = ""
            for word in words:
                if not current:
                    if len(word) > max_chars:
                        # If a single word exceeds the limit, split it.
                        for i in range(0, len(word), max_chars):
                            chunks.append(word[i:i+max_chars])
                        continue
                    else:
                        current = word
                elif len(current) + 1 + len(word) <= max_chars:
                    current += " " + word
                else:
                    chunks.append(current)
                    if len(word) > max_chars:
                        for i in range(0, len(word), max_chars):
                            if i+max_chars < len(word):
                                chunks.append(word[i:i+max_chars])
                            else:
                                current = word[i:]
                    else:
                        current = word
            if current:
                chunks.append(current)
    return chunks

def synthesize_in_chunks(model, text, language, gpt_cond_latent, speaker_embedding, max_chars=250):
    """
    Split text into chunks (each chunk is one or more complete sentences not exceeding max_chars),
    synthesize each chunk, insert 1 second of silence between chunks, and return a list of audio arrays.
    """
    chunks = split_text(text, max_chars)
    print(f"Text split into {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}")
    audios = []
    start_total = time.time()
    # Synthesize each chunk
    for i, chunk in enumerate(chunks):
        start = time.time()
        out = model.inference(
            chunk,
            language,
            gpt_cond_latent,
            speaker_embedding,
            temperature=0.7  # Adjust as needed.
        )
        elapsed = time.time() - start
        print(f"Chunk {i+1}/{len(chunks)} synthesized in {elapsed:.2f}s")
        audios.append(torch.tensor(out["wav"]))
    print(f"Total synthesis time: {time.time()-start_total:.2f}s")
    
    # Insert 1 second of silence (assume sample rate is 24000 Hz)
    sample_rate = 24000
    silence = torch.zeros(sample_rate)
    audio_with_pause = []
    for i, chunk in enumerate(audios):
        audio_with_pause.append(chunk)
        if i < len(audios) - 1:
            audio_with_pause.append(silence)
    return audio_with_pause

def main():
    model = load_model()
    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["/content/Christopher_lee_clean.wav"])
    input_file = "/content/input.txt"
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    print("Running inference in chunks...")
    audio_chunks = synthesize_in_chunks(model, text, "en", gpt_cond_latent, speaker_embedding, max_chars=250)
    full_audio = torch.cat(audio_chunks, dim=0)
    torchaudio.save("/content/xtts.m4a", full_audio.unsqueeze(0), 24000)
    print("Final audio saved as xtts.m4a")

if __name__ == "__main__":
    main()
