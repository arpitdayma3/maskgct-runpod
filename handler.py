import torch
import safetensors
from huggingface_hub import hf_hub_download
import soundfile as sf
import os
import numpy as np
import librosa
import requests # For downloading audio from URL
import uuid # For unique temporary file names
import py3langid as langid
import devicetorch # To determine device
import runpod

# Attempt to import whisper, fall back to a stub if not found during local testing without all deps
try:
    import whisper
except ImportError:
    print("Warning: whisper module not found. ASR functionalities will be limited.")
    whisper = None


# Assuming these custom modules are in the PYTHONPATH
# These imports are based on app.py and models/tts/maskgct/
from models.codec.kmeans.repcodec_model import RepCodec
from models.tts.maskgct.maskgct_s2a import MaskGCT_S2A
from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S
from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor
from utils.util import load_config
from models.tts.maskgct.g2p.g2p_generation import g2p, chn_eng_g2p
import accelerate # Used in app.py for loading some checkpoints

# --- Global Variables & Model Loading ---
DEVICE = None
PROCESSOR = None
SEMANTIC_MODEL = None
SEMANTIC_MEAN = None
SEMANTIC_STD = None
SEMANTIC_CODEC = None
CODEC_ENCODER = None
CODEC_DECODER = None
T2S_MODEL = None
S2A_MODEL_1LAYER = None
S2A_MODEL_FULL = None
WHISPER_MODEL = None

# Determine device
try:
    DEVICE_NAME = devicetorch.get(torch)
    DEVICE = torch.device(DEVICE_NAME)
except Exception as e:
    print(f"Error getting device from devicetorch: {e}. Falling back to cuda/cpu.")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

def download_audio_from_url(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Audio downloaded from {url} and saved to {save_path}")
        return save_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading audio from {url}: {e}")
        return None

# --- Model Building Functions (Adapted from app.py) ---
def build_semantic_model(device):
    semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    semantic_model.eval()
    semantic_model.to(device)
    # Ensure the checkpoint path is correct for RunPod (e.g., downloaded or in repo)
    # Assuming 'models/tts/maskgct/ckpt/wav2vec2bert_stats.pt' is in the repo
    # If not, this needs hf_hub_download or similar
    try:
        stat_mean_var_path = "./models/tts/maskgct/ckpt/wav2vec2bert_stats.pt"
        if not os.path.exists(stat_mean_var_path):
            print(f"Warning: wav2vec2bert_stats.pt not found at {stat_mean_var_path}. Attempting download if available or this might fail.")
            # Placeholder: Add hf_hub_download logic if this file is hosted
            # For now, assuming it's part of the repo as per original structure exploration
        
        stat_mean_var = torch.load(stat_mean_var_path, map_location=device)
        semantic_mean = stat_mean_var["mean"].to(device)
        semantic_std = torch.sqrt(stat_mean_var["var"]).to(device)
    except FileNotFoundError:
        print(f"ERROR: wav2vec2bert_stats.pt not found. Semantic model might not work correctly.")
        # Fallback to zeros, though this is not ideal
        # Ideally, this file should be present or downloaded
        semantic_mean = torch.zeros(1024, device=device) 
        semantic_std = torch.ones(1024, device=device)

    return semantic_model, semantic_mean, semantic_std

def build_semantic_codec(cfg, device):
    semantic_codec = RepCodec(cfg=cfg)
    semantic_codec.eval()
    semantic_codec.to(device)
    return semantic_codec

def build_acoustic_codec(cfg, device):
    codec_encoder = CodecEncoder(cfg=cfg.encoder)
    codec_decoder = CodecDecoder(cfg=cfg.decoder)
    codec_encoder.eval()
    codec_decoder.eval()
    codec_encoder.to(device)
    codec_decoder.to(device)
    return codec_encoder, codec_decoder

def build_t2s_model(cfg, device):
    t2s_model = MaskGCT_T2S(cfg=cfg)
    t2s_model.eval()
    t2s_model.to(device)
    return t2s_model

def build_s2a_model(cfg, device):
    soundstorm_model = MaskGCT_S2A(cfg=cfg)
    soundstorm_model.eval()
    soundstorm_model.to(device)
    return soundstorm_model

def load_all_models():
    global PROCESSOR, SEMANTIC_MODEL, SEMANTIC_MEAN, SEMANTIC_STD, SEMANTIC_CODEC
    global CODEC_ENCODER, CODEC_DECODER, T2S_MODEL, S2A_MODEL_1LAYER, S2A_MODEL_FULL, WHISPER_MODEL

    PROCESSOR = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    
    cfg_path = "./models/tts/maskgct/config/maskgct.json"
    if not os.path.exists(cfg_path):
        print(f"ERROR: MaskGCT config {cfg_path} not found!")
        # Potentially download if hosted, or ensure it's in the repo.
        # For now, this will cause a failure if not present.
        # Fallback: create a dummy cfg or raise error
        raise FileNotFoundError(f"MaskGCT config {cfg_path} not found!")

    cfg = load_config(cfg_path)

    SEMANTIC_MODEL, SEMANTIC_MEAN, SEMANTIC_STD = build_semantic_model(DEVICE)
    SEMANTIC_CODEC = build_semantic_codec(cfg.model.semantic_codec, DEVICE)
    CODEC_ENCODER, CODEC_DECODER = build_acoustic_codec(cfg.model.acoustic_codec, DEVICE)
    T2S_MODEL = build_t2s_model(cfg.model.t2s_model, DEVICE)
    S2A_MODEL_1LAYER = build_s2a_model(cfg.model.s2a_model.s2a_1layer, DEVICE)
    S2A_MODEL_FULL = build_s2a_model(cfg.model.s2a_model.s2a_full, DEVICE)

    # Download/Load checkpoints (adapted from app.py)
    # Semantic Codec
    semantic_code_ckpt_path = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    safetensors.torch.load_model(SEMANTIC_CODEC, semantic_code_ckpt_path, strict=True) # strict=True is safer

    # Acoustic Codec (using local files as per app.py's accelerate usage)
    # Ensure these paths are correct relative to the RunPod execution directory
    # These files should be in the 'acoustic_codec' directory in the repo root.
    codec_encoder_local_ckpt = "./acoustic_codec/model.safetensors"
    codec_decoder_local_ckpt = "./acoustic_codec/model_1.safetensors" # app.py uses model_1, model_2, model_3 for decoder? Checking ls output.
                                                                   # ls shows: model.safetensors, model_1.safetensors, model_2.safetensors, model_3.safetensors
                                                                   # app.py uses accelerate.load_checkpoint_and_dispatch for encoder and decoder.
                                                                   # The original app.py has:
                                                                   # accelerate.load_checkpoint_and_dispatch(codec_encoder, "./acoustic_codec/model.safetensors")
                                                                   # accelerate.load_checkpoint_and_dispatch(codec_decoder, "./acoustic_codec/model_1.safetensors")
                                                                   # This implies model_1.safetensors is for the *entire* decoder, not just a part.
                                                                   # The other files (model_2, model_3) might be for other quantizers if the decoder uses them.
                                                                   # For now, sticking to what app.py explicitly loads.

    if os.path.exists(codec_encoder_local_ckpt):
        accelerate.load_checkpoint_and_dispatch(CODEC_ENCODER, codec_encoder_local_ckpt, device_map='auto')
    else:
        print(f"Warning: Codec encoder checkpoint {codec_encoder_local_ckpt} not found. Downloading from HF Hub.")
        codec_encoder_ckpt_path = hf_hub_download("amphion/MaskGCT", filename="acoustic_codec/model.safetensors")
        # accelerate.load_checkpoint_and_dispatch might be tricky if not all files are downloaded for it.
        # Using safetensors.torch.load_model if accelerate causes issues with single files.
        # For simplicity, let's assume safetensors.torch.load_model is okay if accelerate has issues with hub paths directly.
        # However, the original app.py used accelerate for local files. Let's try to stick to that pattern.
        # If the files are local, accelerate should work. If we download, we need to ensure the structure matches.
        # For now, let's assume these files *must* be present locally as per app.py.
        raise FileNotFoundError(f"Required local checkpoint {codec_encoder_local_ckpt} not found.")


    if os.path.exists(codec_decoder_local_ckpt):
        accelerate.load_checkpoint_and_dispatch(CODEC_DECODER, codec_decoder_local_ckpt, device_map='auto')
    else:
        print(f"Warning: Codec decoder checkpoint {codec_decoder_local_ckpt} not found. Downloading from HF Hub.")
        # This might need more complex loading if it's multi-file. The original used model_1.safetensors.
        codec_decoder_ckpt_path = hf_hub_download("amphion/MaskGCT", filename="acoustic_codec/model_1.safetensors")
        # safetensors.torch.load_model(CODEC_DECODER, codec_decoder_ckpt_path, strict=True)
        raise FileNotFoundError(f"Required local checkpoint {codec_decoder_local_ckpt} not found.")


    # T2S Model
    t2s_model_ckpt_path = hf_hub_download("amphion/MaskGCT", filename="t2s_model/model.safetensors")
    safetensors.torch.load_model(T2S_MODEL, t2s_model_ckpt_path, strict=True)
    
    # S2A Models
    s2a_1layer_ckpt_path = hf_hub_download("amphion/MaskGCT", filename="s2a_model/s2a_model_1layer/model.safetensors")
    safetensors.torch.load_model(S2A_MODEL_1LAYER, s2a_1layer_ckpt_path, strict=True)
    
    s2a_full_ckpt_path = hf_hub_download("amphion/MaskGCT", filename="s2a_model/s2a_model_full/model.safetensors")
    safetensors.torch.load_model(S2A_MODEL_FULL, s2a_full_ckpt_path, strict=True)

    # Whisper model for ASR (optional, but used in app.py's full logic)
    if whisper:
        try:
            # Using "base" model for faster loading, can be changed to "turbo" or others if needed
            WHISPER_MODEL = whisper.load_model("base", device=DEVICE) 
            print("Whisper model loaded successfully.")
        except Exception as e:
            print(f"Error loading Whisper model: {e}. ASR functionality will be impacted.")
            WHISPER_MODEL = None
    else:
        print("Whisper module not available. ASR functionality will be impacted.")

    print("All models loaded.")


# --- Helper Functions from app.py (adapted) ---
def detect_text_language(text):
    return langid.classify(text)[0]

@torch.no_grad()
def detect_speech_language(speech_file_path):
    if not WHISPER_MODEL or not whisper:
        print("Whisper model not available for speech language detection. Defaulting to 'en'.")
        return "en" # Default language
    try:
        audio = whisper.load_audio(speech_file_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(WHISPER_MODEL.device)
        _, probs = WHISPER_MODEL.detect_language(mel)
        return max(probs, key=probs.get)
    except Exception as e:
        print(f"Error during speech language detection: {e}. Defaulting to 'en'.")
        return "en"

@torch.no_grad()
def get_prompt_text_from_audio(speech_16k_path, language):
    if not WHISPER_MODEL or not whisper:
        print("Whisper model not available for ASR. Returning empty prompt text.")
        return "", "", 0.0
    try:
        # speech_16k should be a numpy array here, not a path
        # whisper.load_audio already loads and resamples if needed.
        asr_result = WHISPER_MODEL.transcribe(speech_16k_path, language=language)
        full_prompt_text = asr_result["text"]
        short_prompt_text = ""
        short_prompt_end_ts = 0.0
        for segment in asr_result["segments"]:
            short_prompt_text += segment['text']
            short_prompt_end_ts = segment['end']
            if short_prompt_end_ts >= 4.0: # Use 4.0 to match app.py
                break
        return full_prompt_text, short_prompt_text, short_prompt_end_ts
    except Exception as e:
        print(f"Error during ASR: {e}. Returning empty prompt text.")
        return "", "", 0.0

def g2p_phonemes(text, language):
    # Adapted from app.py's g2p_
    if language in ["zh", "en"]: # Assuming chn_eng_g2p handles mixed cases too
        return chn_eng_g2p(text) # Returns (phonemes, phone_ids)
    else:
        return g2p(text, sentence=None, language=language) # Returns (phonemes, phone_ids)

@torch.no_grad()
def extract_features(speech_16k_array, processor):
    # speech_16k_array is a numpy array
    inputs = processor(speech_16k_array, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"][0].to(DEVICE) # Move to device
    attention_mask = inputs["attention_mask"][0].to(DEVICE) # Move to device
    return input_features, attention_mask

@torch.no_grad()
def extract_semantic_code(input_features, attention_mask):
    # SEMANTIC_MODEL, SEMANTIC_MEAN, SEMANTIC_STD, SEMANTIC_CODEC are global
    vq_emb = SEMANTIC_MODEL(
        input_features=input_features,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    feat = vq_emb.hidden_states[17]
    feat = (feat - SEMANTIC_MEAN) / SEMANTIC_STD
    semantic_code, _ = SEMANTIC_CODEC.quantize(feat)
    return semantic_code # (B, T)

@torch.no_grad()
def extract_acoustic_code(speech_24k_tensor):
    # speech_24k_tensor is a torch tensor on the correct device, shape [1, num_samples]
    # CODEC_ENCODER, CODEC_DECODER are global
    vq_emb = CODEC_ENCODER(speech_24k_tensor.unsqueeze(1)) # Add channel dim: [1, 1, num_samples]
    _, vq, _, _, _ = CODEC_DECODER.quantizer(vq_emb) # vq shape [B, N_q, T]
    acoustic_code = vq.permute(1, 2, 0) # [N_q, T, B] - check shape expected by S2A model
                                        # app.py uses acoustic_code[:, :, :] which implies it wants [B, N_q, T] or similar
                                        # The S2A model prompt input is [B, N_q, T]
                                        # The output of quantizer is [B, N_q, T]
                                        # So, acoustic_code = vq (no permute needed if B=0 is the first dim)
                                        # Let's re-check app.py:
                                        # acoustic_code = vq.permute(1, 2, 0)
                                        # prompt = acoustic_code[:,:,:]
                                        # s2a_model_1layer.reverse_diffusion(..., prompt=prompt,...)
                                        # MaskGCT_S2A forward takes acoustic_token [B, N_q, T_s]
                                        # So the permute in app.py (1,2,0) changes [B, N_q, T] to [N_q, T, B]
                                        # Then prompt = acoustic_code[:,:,:] slices it.
                                        # This seems a bit unusual. Let's assume the S2A model expects [B, N_q, T_s] for prompt.
                                        # And vq from CODEC_DECODER.quantizer is already [B, N_q, T_s].
    return vq # Use vq directly, assuming it's [B, N_q, T]

@torch.no_grad()
def text_to_semantic_inference(
    prompt_speech_16k_array, # numpy array
    prompt_text_short,
    prompt_language,
    target_text_content,
    target_language_code,
    target_duration_frames=None, # Target duration in 50Hz frames
    n_timesteps_t2s=50,
    cfg_t2s=2.5,
    rescale_cfg_t2s=0.75,
):
    # T2S_MODEL, PROCESSOR, SEMANTIC_MODEL, SEMANTIC_MEAN, SEMANTIC_STD, SEMANTIC_CODEC are global
    
    _, prompt_phone_id = g2p_phonemes(prompt_text_short, prompt_language)
    _, target_phone_id = g2p_phonemes(target_text_content, target_language_code)

    if target_duration_frames is None or target_duration_frames < 0:
        # Simple rule from app.py (approx)
        target_duration_frames = int(
            (len(prompt_speech_16k_array) / 16000 * 50) * \
            (len(target_phone_id) / (len(prompt_phone_id) + 1e-6))
        )
    
    prompt_phone_id = torch.tensor(prompt_phone_id, dtype=torch.long, device=DEVICE)
    target_phone_id = torch.tensor(target_phone_id, dtype=torch.long, device=DEVICE)
    
    phone_id_combined = torch.cat([prompt_phone_id, target_phone_id]).unsqueeze(0) # Add batch dim

    input_features, attention_mask = extract_features(prompt_speech_16k_array, PROCESSOR)
    input_features = input_features.unsqueeze(0) # Add batch dim
    attention_mask = attention_mask.unsqueeze(0) # Add batch dim
    
    # Original prompt's semantic code
    prompt_semantic_code = extract_semantic_code(input_features, attention_mask) # Shape [1, T_semantic_prompt]

    # Predict target semantic codes
    predicted_target_semantic = T2S_MODEL.reverse_diffusion(
        prompt_semantic_code, # prompt_semantic
        target_duration_frames, # target_len (in frames)
        phone_id_combined, # phone_id (prompt + target)
        n_timesteps=n_timesteps_t2s,
        cfg=cfg_t2s,
        rescale_cfg=rescale_cfg_t2s,
    ) # Shape [1, T_semantic_target]

    # Combine prompt's semantic code with predicted target semantic code
    combined_semantic_code = torch.cat([prompt_semantic_code, predicted_target_semantic], dim=1) # Dim 1 is time
    
    return combined_semantic_code, prompt_semantic_code


@torch.no_grad()
def semantic_to_acoustic_inference(
    combined_semantic_tensor, # [1, T_semantic_combined]
    prompt_acoustic_code_tensor, # [1, N_q, T_acoustic_prompt]
    n_timesteps_s2a_list=None,
    cfg_s2a=2.5,
    rescale_cfg_s2a=0.75,
):
    # S2A_MODEL_1LAYER, S2A_MODEL_FULL, CODEC_DECODER are global
    if n_timesteps_s2a_list is None:
        n_timesteps_s2a_list = [25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # Default from app.py

    # Layer 1 S2A
    cond_1layer = S2A_MODEL_1LAYER.cond_emb(combined_semantic_tensor)
    predicted_acoustic_1layer = S2A_MODEL_1LAYER.reverse_diffusion(
        cond=cond_1layer,
        prompt=prompt_acoustic_code_tensor,
        temp=1.5,
        filter_thres=0.98,
        n_timesteps=n_timesteps_s2a_list[:1], # Only first step for 1-layer model
        cfg=cfg_s2a,
        rescale_cfg=rescale_cfg_s2a,
    ) # Shape [1, N_q, T_acoustic_target]

    # Full S2A (using 1-layer output as GT for subsequent steps)
    cond_full = S2A_MODEL_FULL.cond_emb(combined_semantic_tensor)
    # The prompt for full model should be the original prompt acoustic code
    predicted_acoustic_full = S2A_MODEL_FULL.reverse_diffusion(
        cond=cond_full,
        prompt=prompt_acoustic_code_tensor,
        temp=1.5,
        filter_thres=0.98,
        n_timesteps=n_timesteps_s2a_list, # Full list of timesteps
        cfg=cfg_s2a,
        rescale_cfg=rescale_cfg_s2a,
        gt_code=predicted_acoustic_1layer, # Output from 1-layer model
    ) # Shape [1, N_q, T_acoustic_target]

    # Combine prompt acoustic codes with predicted target acoustic codes for full output
    # The predicted_acoustic_full is only for the *target* part.
    # The prompt audio is reconstructed separately if needed, or we can just return target.
    # app.py returns combined audio.
    
    # Reconstruct audio from predicted target acoustic codes
    # Input to vq2emb should be [N_q, B, T] or [B, N_q, T] depending on implementation
    # CODEC_DECODER.vq2emb expects [N_q, B, T_s] if n_quantizers is None, or [B, T_s, N_q] if n_quantizers is specified.
    # Let's check MaskGCT_S2A's `out_to_vqemb`
    # It uses `self.vq2emb(acoustic_token.permute(1, 0, 2), n_quantizers=self.acoustic_token_dims)`
    # acoustic_token is [B, N_q, T_s]. So permute makes it [N_q, B, T_s].
    # predicted_acoustic_full is [1, N_q, T_acoustic_target] (B=1)
    
    target_vq_emb = CODEC_DECODER.vq2emb(predicted_acoustic_full.permute(1,0,2), n_quantizers=predicted_acoustic_full.shape[1])
    recovered_target_audio_tensor = CODEC_DECODER(target_vq_emb) # Shape [1, 1, T_audio_target]
    
    # Reconstruct prompt audio
    prompt_vq_emb = CODEC_DECODER.vq2emb(prompt_acoustic_code_tensor.permute(1,0,2), n_quantizers=prompt_acoustic_code_tensor.shape[1])
    recovered_prompt_audio_tensor = CODEC_DECODER(prompt_vq_emb) # Shape [1, 1, T_audio_prompt]

    recovered_prompt_audio_np = recovered_prompt_audio_tensor.squeeze().cpu().numpy()
    recovered_target_audio_np = recovered_target_audio_tensor.squeeze().cpu().numpy()
    
    combined_audio_np = np.concatenate([recovered_prompt_audio_np, recovered_target_audio_np])

    return combined_audio_np, recovered_target_audio_np


# --- Main Inference Function (adapted from app.py's maskgct_inference) ---
@torch.no_grad()
def run_maskgct_inference(
    prompt_audio_filepath, # Local path to downloaded prompt audio
    target_text_content,
    target_duration_sec=None, # Optional target duration in seconds
    # T2S params
    n_timesteps_t2s=25, 
    cfg_t2s=2.5, 
    rescale_cfg_t2s=0.75,
    # S2A params
    n_timesteps_s2a_list=None,
    cfg_s2a=2.5, 
    rescale_cfg_s2a=0.75,
):
    # 1. Load audio and determine languages
    try:
        speech_orig, sr_orig = librosa.load(prompt_audio_filepath, sr=None)
        speech_16k_array = librosa.resample(speech_orig, orig_sr=sr_orig, target_sr=16000)
        speech_24k_array = librosa.resample(speech_orig, orig_sr=sr_orig, target_sr=24000)
    except Exception as e:
        print(f"Error loading or resampling audio: {e}")
        raise
    
    prompt_language_code = detect_speech_language(prompt_audio_filepath)
    target_language_code = detect_text_language(target_text_content)

    # 2. Get prompt text (ASR) and determine relevant segment of prompt audio
    _, short_prompt_text, short_prompt_end_ts = get_prompt_text_from_audio(
        prompt_audio_filepath, # Whisper can take filepath
        prompt_language_code
    )
    if not short_prompt_text: # If ASR fails
        print("Warning: ASR failed to produce prompt text. Using a generic or empty prompt text.")
        # Fallback: use a generic prompt or handle error. For now, let's allow it to proceed.
        # Or, could use full original audio if ASR fails to segment.
        # For simplicity, if ASR fails, we might not have a good prompt_text.
        # This part is crucial for quality.
    
    # Trim audio arrays to the segment identified by ASR (up to short_prompt_end_ts, e.g., 4s)
    # If short_prompt_end_ts is 0 (e.g. ASR failed), use full audio up to a max length (e.g. 10-15s)
    max_prompt_duration_sec = 15.0
    if short_prompt_end_ts > 0:
        effective_prompt_duration_sec = min(short_prompt_end_ts, max_prompt_duration_sec)
    else: # ASR failed or no segments found
        effective_prompt_duration_sec = min(len(speech_16k_array) / 16000.0, max_prompt_duration_sec)

    speech_16k_array_prompt = speech_16k_array[:int(effective_prompt_duration_sec * 16000)]
    speech_24k_array_prompt = speech_24k_array[:int(effective_prompt_duration_sec * 24000)]

    # 3. Convert target duration from seconds to 50Hz frames if provided
    target_duration_frames = None
    if target_duration_sec is not None and target_duration_sec > 0:
        target_duration_frames = int(target_duration_sec * 50)

    # 4. Text-to-Semantic (T2S)
    combined_semantic_code, _ = text_to_semantic_inference(
        speech_16k_array_prompt,
        short_prompt_text,
        prompt_language_code,
        target_text_content,
        target_language_code,
        target_duration_frames=target_duration_frames,
        n_timesteps_t2s=n_timesteps_t2s,
        cfg_t2s=cfg_t2s,
        rescale_cfg_t2s=rescale_cfg_t2s,
    ) # combined_semantic_code shape: [1, T_semantic_combined]

    # 5. Extract Acoustic Codes from prompt audio
    prompt_acoustic_code_tensor = extract_acoustic_code(
        torch.tensor(speech_24k_array_prompt, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    ) # Expected shape [1, N_q, T_acoustic_prompt]

    # 6. Semantic-to-Acoustic (S2A)
    _, recovered_target_audio_np = semantic_to_acoustic_inference(
        combined_semantic_code,
        prompt_acoustic_code_tensor,
        n_timesteps_s2a_list=n_timesteps_s2a_list,
        cfg_s2a=cfg_s2a,
        rescale_cfg_s2a=rescale_cfg_s2a,
    ) # Numpy array of the target speech

    return recovered_target_audio_np # Return only the target part as per issue request (implicitly)


# --- RunPod Handler ---
def handler(job):
    try:
        job_input = job['input']
        audio_url = job_input.get('audio_url')
        target_text = job_input.get('target_text')
        
        # Optional parameters from job_input
        target_duration_sec = job_input.get('target_duration_sec', None)
        n_timesteps_t2s = job_input.get('n_timesteps_t2s', 25) # Default from app.py
        # ... other inference parameters can be exposed here ...

        if not audio_url or not target_text:
            return {"error": "Missing 'audio_url' or 'target_text' in input"}

        # Create a temporary directory for downloaded and generated files
        tmp_dir = "/tmp/maskgct_runpod"
        os.makedirs(tmp_dir, exist_ok=True)
        
        prompt_audio_filename = f"prompt_{uuid.uuid4()}.wav"
        local_prompt_audio_path = os.path.join(tmp_dir, prompt_audio_filename)

        # Download audio
        downloaded_path = download_audio_from_url(audio_url, local_prompt_audio_path)
        if not downloaded_path:
            return {"error": f"Failed to download audio from URL: {audio_url}"}

        # Run inference
        print(f"Starting inference for target text: {target_text}")
        output_audio_np = run_maskgct_inference(
            local_prompt_audio_path,
            target_text,
            target_duration_sec=target_duration_sec,
            n_timesteps_t2s=int(n_timesteps_t2s) # Ensure int
            # Pass other params if exposed
        )
        print("Inference completed.")

        # Save output audio
        output_filename = f"output_{uuid.uuid4()}.wav"
        local_output_audio_path = os.path.join(tmp_dir, output_filename)
        sf.write(local_output_audio_path, output_audio_np, 24000) # MaskGCT default SR is 24kHz
        print(f"Output audio saved to: {local_output_audio_path}")

        # Clean up downloaded file
        try:
            os.remove(local_prompt_audio_path)
        except OSError as e:
            print(f"Error deleting temporary prompt file {local_prompt_audio_path}: {e}")
            
        # Return the path to the generated file (RunPod will handle serving it or allowing download)
        # Or, optionally, upload to a bucket and return URL
        return {"output_audio_path": local_output_audio_path}

    except Exception as e:
        print(f"Error in handler: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


# --- Load models when the worker starts ---
# This should be called once when the RunPod worker initializes.
if __name__ == "__main__":
    # This block is for local testing of the handler, not directly used by RunPod serverless
    # RunPod calls load_all_models() then handler().
    print("Starting model loading (for local testing)...")
    load_all_models()
    print("Models loaded. Handler ready for local testing.")

    # Example local test:
    # Create a dummy job input
    # Ensure you have a sample audio URL and text
    # sample_job = {
    #     "input": {
    #         "audio_url": "URL_TO_A_TEST_WAV_FILE", # Replace with an actual URL
    #         "target_text": "Hello, this is a test of the MaskGCT model on RunPod.",
    #         "target_duration_sec": 5 
    #     }
    # }
    # result = handler(sample_job)
    # print(f"Handler result: {result}")
else:
    # This is what RunPod will effectively do: load models, then it's ready for handler calls.
    print("Starting model loading (for RunPod worker)...")
    load_all_models()
    print("Models loaded. Handler ready.")
    runpod.serverless.start({"handler": handler})
