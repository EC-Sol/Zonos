import torch
import torchaudio
import gradio as gr
from os import getenv
import os
import nltk
import tempfile
from pydub import AudioSegment

from zonos.model import Zonos, DEFAULT_BACKBONE_CLS as ZonosBackbone
from zonos.conditioning import make_cond_dict, supported_language_codes
from zonos.utils import DEFAULT_DEVICE as device

CURRENT_MODEL_TYPE = None
CURRENT_MODEL = None

SPEAKER_EMBEDDING = None
SPEAKER_AUDIO_PATH = None


def load_model_if_needed(model_choice: str):
    global CURRENT_MODEL_TYPE, CURRENT_MODEL
    if CURRENT_MODEL_TYPE != model_choice:
        if CURRENT_MODEL is not None:
            del CURRENT_MODEL
            torch.cuda.empty_cache()
        print(f"Loading {model_choice} model...")
        CURRENT_MODEL = Zonos.from_pretrained(model_choice, device=device)
        CURRENT_MODEL.requires_grad_(False).eval()
        CURRENT_MODEL_TYPE = model_choice
        print(f"{model_choice} model loaded successfully!")
    return CURRENT_MODEL


def update_ui(model_choice):
    """
    Dynamically show/hide UI elements based on the model's conditioners.
    We do NOT display 'language_id' or 'ctc_loss' even if they exist in the model.
    """
    model = load_model_if_needed(model_choice)
    cond_names = [c.name for c in model.prefix_conditioner.conditioners]
    print("Conditioners in this model:", cond_names)

    text_update = gr.update(visible=("espeak" in cond_names))
    language_update = gr.update(visible=("espeak" in cond_names))
    speaker_audio_update = gr.update(visible=("speaker" in cond_names))
    prefix_audio_update = gr.update(visible=True)
    emotion1_update = gr.update(visible=("emotion" in cond_names))
    emotion2_update = gr.update(visible=("emotion" in cond_names))
    emotion3_update = gr.update(visible=("emotion" in cond_names))
    emotion4_update = gr.update(visible=("emotion" in cond_names))
    emotion5_update = gr.update(visible=("emotion" in cond_names))
    emotion6_update = gr.update(visible=("emotion" in cond_names))
    emotion7_update = gr.update(visible=("emotion" in cond_names))
    emotion8_update = gr.update(visible=("emotion" in cond_names))
    vq_single_slider_update = gr.update(visible=("vqscore_8" in cond_names))
    fmax_slider_update = gr.update(visible=("fmax" in cond_names))
    pitch_std_slider_update = gr.update(visible=("pitch_std" in cond_names))
    speaking_rate_slider_update = gr.update(visible=("speaking_rate" in cond_names))
    dnsmos_slider_update = gr.update(visible=("dnsmos_ovrl" in cond_names))
    speaker_noised_checkbox_update = gr.update(visible=("speaker_noised" in cond_names))
    unconditional_keys_update = gr.update(
        choices=[name for name in cond_names if name not in ("espeak", "language_id")]
    )

    return (
        text_update,
        language_update,
        speaker_audio_update,
        prefix_audio_update,
        emotion1_update,
        emotion2_update,
        emotion3_update,
        emotion4_update,
        emotion5_update,
        emotion6_update,
        emotion7_update,
        emotion8_update,
        vq_single_slider_update,
        fmax_slider_update,
        pitch_std_slider_update,
        speaking_rate_slider_update,
        dnsmos_slider_update,
        speaker_noised_checkbox_update,
        unconditional_keys_update,
    )


def generate_audio(
        model_choice,
        text,
        language,
        speaker_audio,
        prefix_audio,
        e1,
        e2,
        e3,
        e4,
        e5,
        e6,
        e7,
        e8,
        vq_single,
        fmax,
        pitch_std,
        speaking_rate,
        dnsmos_ovrl,
        speaker_noised,
        cfg_scale,
        top_p,
        top_k,
        min_p,
        linear,
        confidence,
        quadratic,
        seed,
        randomize_seed,
        unconditional_keys,
        progress=gr.Progress(),
):
    """
    Generates audio based on the provided UI parameters.
    We do NOT use language_id or ctc_loss even if the model has them.
    """

    selected_model = load_model_if_needed(model_choice)

    speaker_noised_bool = bool(speaker_noised)
    fmax = float(fmax)
    pitch_std = float(pitch_std)
    speaking_rate = float(speaking_rate)
    dnsmos_ovrl = float(dnsmos_ovrl)
    cfg_scale = float(cfg_scale)
    top_p = float(top_p)
    top_k = int(top_k)
    min_p = float(min_p)
    linear = float(linear)
    confidence = float(confidence)
    quadratic = float(quadratic)
    seed = int(seed)
    max_new_tokens = 86 * 60 * 60

    # This is a bit ew, but works for now.
    global SPEAKER_AUDIO_PATH, SPEAKER_EMBEDDING

    if randomize_seed:
        seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
    torch.manual_seed(seed)

    if speaker_audio is not None and "speaker" not in unconditional_keys:
        if speaker_audio != SPEAKER_AUDIO_PATH:
            print("Recomputed speaker embedding")
            wav, sr = torchaudio.load(speaker_audio)
            SPEAKER_EMBEDDING = selected_model.make_speaker_embedding(wav, sr)
            SPEAKER_EMBEDDING = SPEAKER_EMBEDDING.to(device, dtype=torch.bfloat16)
            SPEAKER_AUDIO_PATH = speaker_audio

    # Initial audio_prefix_codes setup
    processed_prefix_audio = None
    if prefix_audio is not None:
        wav_prefix, sr_prefix = torchaudio.load(prefix_audio)
        wav_prefix = wav_prefix.mean(0, keepdim=True) # Ensure mono
        # preprocess expects (wav: Tensor, sr: int) -> wav (Tensor), sr (int)
        # input wav can be [B, T] or [T]. Output is [C, T_proc]
        processed_wav_prefix, _ = selected_model.autoencoder.preprocess(wav_prefix, sr_prefix)
        processed_wav_prefix = processed_wav_prefix.to(device, dtype=torch.float32)
        # encode expects [B, C, T_proc]
        processed_prefix_audio = selected_model.autoencoder.encode(processed_wav_prefix.unsqueeze(0))

    current_prefix_codes = processed_prefix_audio # Initialize for chunking

    emotion_tensor = torch.tensor(list(map(float, [e1, e2, e3, e4, e5, e6, e7, e8])), device=device)
    vq_val = float(vq_single)
    vq_tensor = torch.tensor([vq_val] * 8, device=device).unsqueeze(0)

    output_audio_files = []
    CHAR_LIMIT = 250  # Max characters per chunk for NLTK splitting

    if len(text) > CHAR_LIMIT:
        sentences = nltk.sent_tokenize(text)
        current_chunk = ""
        chunk_count = 0
        total_chunks = len(sentences)  # Approximate, as sentences are combined

        for i, sentence in enumerate(sentences):
            progress((i, total_chunks), desc=f"Processing text chunks... {total_chunks}")
            if len(current_chunk) + len(sentence) + 1 > CHAR_LIMIT and current_chunk:
                chunk_count += 1
                progress((i, total_chunks), desc=f"Processing text chunks... {chunk_count} / {int(len(text) / CHAR_LIMIT)}")
                print(f'current_chunk: {current_chunk}')

                # Generate audio for current_chunk
                cond_dict_chunk = make_cond_dict(
                    text=current_chunk.strip(),
                    language=language,
                    speaker=SPEAKER_EMBEDDING,
                    emotion=emotion_tensor,
                    vqscore_8=vq_tensor,
                    fmax=fmax,
                    pitch_std=pitch_std,
                    speaking_rate=speaking_rate,
                    dnsmos_ovrl=dnsmos_ovrl,
                    speaker_noised=speaker_noised_bool,
                    device=device,
                    unconditional_keys=unconditional_keys,
                )
                conditioning_chunk = selected_model.prepare_conditioning(cond_dict_chunk)

                # current_prefix_codes is now managed iteratively
                # It's initialized with processed_prefix_audio (user prefix) before the loop
                # And updated from the previous chunk's output at the end of each iteration

                codes_chunk = selected_model.generate(
                    prefix_conditioning=conditioning_chunk,
                    audio_prefix_codes=current_prefix_codes, # Uses user prefix for 1st chunk, then previous chunk's output
                    max_new_tokens=max_new_tokens,
                    cfg_scale=cfg_scale,
                    batch_size=1,
                    sampling_params=dict(top_p=top_p, top_k=top_k, min_p=min_p, linear=linear, conf=confidence,
                                         quad=quadratic),
                    # No per-token callback for chunked generation to simplify
                )
                wav_out_chunk = selected_model.autoencoder.decode(codes_chunk).cpu().detach()
                sr_out_chunk = selected_model.autoencoder.sampling_rate
                if wav_out_chunk.dim() == 2 and wav_out_chunk.size(0) > 1: # Ensure mono, [1, T]
                    wav_out_chunk = wav_out_chunk[0:1, :]
                elif wav_out_chunk.dim() == 1: # if it's [T]
                    wav_out_chunk = wav_out_chunk.unsqueeze(0) # make it [1, T]

                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False,
                                                        dir="./temp_audio")
                os.makedirs("./temp_audio", exist_ok=True)
                torchaudio.save(temp_file.name, wav_out_chunk.squeeze().unsqueeze(0), sr_out_chunk) # Save still expects [1, T] or [T]
                output_audio_files.append(temp_file.name)
                temp_file.close()

                # Prepare prefix codes for the next iteration from the current chunk's output
                # wav_out_chunk is on CPU, shape [1, num_samples]
                wav_for_next_prefix = wav_out_chunk.to(device) # Move to device
                # preprocess expects (wav: Tensor, sr: int) -> wav (Tensor), sr (int)
                # input wav can be [B, T] or [T]. Output is [C, T_proc] e.g. [1, T_proc]
                processed_wav_for_next_prefix, _ = selected_model.autoencoder.preprocess(wav_for_next_prefix, sr_out_chunk)
                # encode expects [B, C, T_proc]
                current_prefix_codes = selected_model.autoencoder.encode(processed_wav_for_next_prefix.unsqueeze(0))

                current_chunk = sentence + " "
            else:
                current_chunk += sentence + " "

        # Process the last remaining chunk
        if current_chunk.strip():
            chunk_count += 1
            progress((total_chunks, total_chunks),
                     desc="Processing final chunk...")  # Update progress for the last chunk
            cond_dict_chunk = make_cond_dict(
                text=current_chunk.strip(),
                language=language,
                speaker=SPEAKER_EMBEDDING,
                emotion=emotion_tensor,
                vqscore_8=vq_tensor,
                fmax=fmax,
                pitch_std=pitch_std,
                speaking_rate=speaking_rate,
                dnsmos_ovrl=dnsmos_ovrl,
                speaker_noised=speaker_noised_bool,
                device=device,
                unconditional_keys=unconditional_keys,
            )
            conditioning_chunk = selected_model.prepare_conditioning(cond_dict_chunk)
            # current_prefix_codes will hold the prefix from the last processed chunk in the loop
            codes_chunk = selected_model.generate(
                prefix_conditioning=conditioning_chunk,
                audio_prefix_codes=current_prefix_codes, # Use the prefix from the previous chunk
                max_new_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                batch_size=1,
                sampling_params=dict(top_p=top_p, top_k=top_k, min_p=min_p, linear=linear, conf=confidence,
                                     quad=quadratic),
            )
            wav_out_chunk = selected_model.autoencoder.decode(codes_chunk).cpu().detach()
            sr_out_chunk = selected_model.autoencoder.sampling_rate
            if wav_out_chunk.dim() == 2 and wav_out_chunk.size(0) > 1: # Ensure mono, [1, T]
                wav_out_chunk = wav_out_chunk[0:1, :]
            elif wav_out_chunk.dim() == 1: # if it's [T]
                wav_out_chunk = wav_out_chunk.unsqueeze(0) # make it [1, T]


            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="./temp_audio")
            os.makedirs("./temp_audio", exist_ok=True)
            torchaudio.save(temp_file.name, wav_out_chunk.squeeze().unsqueeze(0), sr_out_chunk) # Save still expects [1, T] or [T]
            output_audio_files.append(temp_file.name)
            temp_file.close()
            # No need to update current_prefix_codes here as this is the last chunk

        return (output_audio_files, seed)

    else:
        # Original logic for single audio file (text <= CHAR_LIMIT)
        # It should use the initial processed_prefix_audio if provided
        current_prefix_codes = processed_prefix_audio # Ensure this is used for non-chunked generation too
        cond_dict = make_cond_dict(
            text=text,
            language=language,
            speaker=SPEAKER_EMBEDDING,
            emotion=emotion_tensor,
            vqscore_8=vq_tensor,
            fmax=fmax,
            pitch_std=pitch_std,
            speaking_rate=speaking_rate,
            dnsmos_ovrl=dnsmos_ovrl,
            speaker_noised=speaker_noised_bool,
            device=device,
            unconditional_keys=unconditional_keys,
        )
        conditioning = selected_model.prepare_conditioning(cond_dict)

        estimated_generation_duration = 30 * len(text) / 400
        estimated_total_steps = int(estimated_generation_duration * 86)

        def update_progress(_frame: torch.Tensor, step: int, _total_steps: int) -> bool:
            progress((step, estimated_total_steps), desc="Generating audio...")
            return True

        codes = selected_model.generate(
            prefix_conditioning=conditioning,
            audio_prefix_codes=current_prefix_codes,  # Use the (potentially user-provided) prefix audio
            max_new_tokens=max_new_tokens,
            cfg_scale=cfg_scale,
            batch_size=1,
            sampling_params=dict(top_p=top_p, top_k=top_k, min_p=min_p, linear=linear, conf=confidence, quad=quadratic),
            callback=update_progress,
        )
        wav_out = selected_model.autoencoder.decode(codes).cpu().detach()
        sr_out = selected_model.autoencoder.sampling_rate
        if wav_out.dim() == 2 and wav_out.size(0) > 1:
            wav_out = wav_out[0:1, :]

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="./temp_audio")
        os.makedirs("./temp_audio", exist_ok=True)
        torchaudio.save(temp_file.name, wav_out.squeeze().unsqueeze(0), sr_out)
        output_audio_files.append(temp_file.name)
        temp_file.close()

    # Merging logic for audio chunks
    final_wav_path = None  # For clarity, though output_audio_files is the key return component
    if output_audio_files:
        if len(output_audio_files) > 1:
            progress(0, 1, desc="Merging audio chunks...")  # Indicate merging start
            try:
                os.makedirs("./temp_audio", exist_ok=True)

                # Load the first audio file to get its properties
                first_waveform, sr = torchaudio.load(output_audio_files[0])
                first_waveform = first_waveform.to(device)  # Ensure it's on the target device
                audio_tensors = [first_waveform]

                silence_duration_ms = 100
                silence_frames = int(sr * (silence_duration_ms / 1000.0))
                # Ensure silence tensor matches the first waveform's device and channel count
                silence_tensor = torch.zeros((first_waveform.shape[0], silence_frames), dtype=first_waveform.dtype,
                                             device=first_waveform.device)

                for i in range(1, len(output_audio_files)):
                    file_path = output_audio_files[i]
                    audio_tensors.append(silence_tensor)
                    waveform, current_sr = torchaudio.load(file_path)
                    waveform = waveform.to(first_waveform.device)  # Move to the same device

                    if current_sr != sr:
                        print(
                            f"Warning: Sample rate mismatch. Expected {sr}, got {current_sr} for {file_path}. Resampling.")
                        waveform = torchaudio.transforms.Resample(current_sr, sr, device=first_waveform.device)(
                            waveform)

                    # Ensure channel count matches the first waveform (e.g., by taking mean for mono, or duplicating for stereo)
                    if waveform.shape[0] != first_waveform.shape[0]:
                        if first_waveform.shape[0] == 1:  # Target is mono
                            waveform = waveform.mean(dim=0, keepdim=True)
                        elif waveform.shape[0] == 1 and first_waveform.shape[0] > 1:  # Source is mono, target stereo
                            waveform = waveform.expand(first_waveform.shape[0], -1)
                        else:  # Fallback: convert current to mono if complex mismatch
                            print(f"Complex channel mismatch for {file_path}, converting to mono.")
                            waveform = waveform.mean(dim=0, keepdim=True)
                            # If first_waveform was stereo, convert it to mono as well for consistency
                            if first_waveform.shape[0] > 1:
                                print("Converting first_waveform to mono due to channel mismatch.")
                                first_waveform = first_waveform.mean(dim=0, keepdim=True)
                                audio_tensors[0] = first_waveform  # Update the stored tensor
                                # Recreate silence tensor for mono
                                silence_tensor = torch.zeros((1, silence_frames), dtype=first_waveform.dtype,
                                                             device=first_waveform.device)
                                audio_tensors[i * 2 - 1] = silence_tensor  # Update previous silence if it was stereo

                    audio_tensors.append(waveform)

                merged_waveform = torch.cat(audio_tensors, dim=1)

                merged_temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="./temp_audio")
                torchaudio.save(merged_temp_file.name, merged_waveform.cpu(), sr)  # Save needs CPU tensor
                final_wav_path = merged_temp_file.name
                merged_temp_file.close()

                # Delete original chunk files
                original_chunk_paths = list(output_audio_files)  # Copy before modifying
                for file_path in original_chunk_paths:
                    try:
                        if os.path.exists(file_path):  # Check if file still exists
                            os.remove(file_path)
                    except Exception as e:
                        print(f"Error deleting chunk file {file_path}: {e}")

                output_audio_files = [final_wav_path]  # Update to contain only the merged file
                progress(1, 1, desc="Audio chunks merged.")
            except Exception as e:
                print(f"Error during audio merging: {e}")
                # If merging fails, output_audio_files will retain original chunks.
                # This might lead to multiple files in output if not handled by subsequent steps.
        elif output_audio_files:  # Exactly one file
            final_wav_path = output_audio_files[0]
            # output_audio_files already contains the correct path, so no change needed to the list itself.

    # If final_wav_path is None here, it means no audio was generated or an error occurred before this stage.
    # The function will return output_audio_files which might be empty or contain original chunks if merging failed.

    # MP3 Compression
    if output_audio_files and output_audio_files[0].endswith(".wav"):  # Only process if we have a WAV file
        wav_file_path = output_audio_files[0]
        mp3_file_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir="./temp_audio").name

        try:
            # Attempt 1: Using torchaudio
            progress(0, 1, desc="Compressing to MP3 (torchaudio)...")
            waveform, sr = torchaudio.load(wav_file_path)
            torchaudio.save(mp3_file_path, waveform, sr, format="mp3")
            print(f"Successfully compressed to MP3 using torchaudio: {mp3_file_path}")
            output_audio_files = [mp3_file_path]
            progress(1, 1, desc="MP3 compression successful (torchaudio).")
            try:
                if os.path.exists(wav_file_path): os.remove(wav_file_path)
            except OSError as e:
                print(f"Error deleting temporary WAV file {wav_file_path}: {e}")
        except Exception as e_torchaudio:
            print(f"torchaudio MP3 conversion failed: {e_torchaudio}. Trying with pydub.")
            progress(0, 1, desc="Compressing to MP3 (pydub)...")
            # Attempt 2: Using pydub
            try:
                audio = AudioSegment.from_wav(wav_file_path)
                audio.export(mp3_file_path, format="mp3")
                print(f"Successfully compressed to MP3 using pydub: {mp3_file_path}")
                output_audio_files = [mp3_file_path]
                progress(1, 1, desc="MP3 compression successful (pydub).")
                try:
                    if os.path.exists(wav_file_path): os.remove(wav_file_path)
                except OSError as e_remove_wav:
                    print(f"Error deleting temporary WAV file {wav_file_path} after pydub conversion: {e_remove_wav}")
            except Exception as e_pydub:
                print(f"pydub MP3 conversion also failed: {e_pydub}")
                progress(1, 1, desc="MP3 compression failed.")
                print("MP3 conversion failed. Returning original WAV path.")
                # output_audio_files already contains the WAV path
                if os.path.exists(mp3_file_path):  # Clean up empty/failed mp3 file
                    try:
                        # Check size because NamedTemporaryFile creates an empty file
                        if os.path.getsize(mp3_file_path) == 0:
                            os.remove(mp3_file_path)
                    except OSError:
                        pass  # Ignore if deletion fails

    return (output_audio_files, seed)


def build_interface():
    supported_models = []

    # if "transformer" in ZonosBackbone.supported_architectures:
    #     supported_models.append("Zyphra/Zonos-v0.1-transformer")

    if "hybrid" in ZonosBackbone.supported_architectures:
        supported_models.append("Zyphra/Zonos-v0.1-hybrid")

    else:
        print(
            "| The current ZonosBackbone does not support the hybrid architecture, meaning only the transformer model will be available in the model selector.\n"
            "| This probably means the mamba-ssm library has not been installed."
        )

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                model_choice = gr.Dropdown(
                    choices=supported_models,
                    value=supported_models[0],
                    label="Zonos Model Type",
                    info="Select the model variant to use.",
                )
                text = gr.Textbox(
                    label="Text to Synthesize",
                    value="Zonos uses eSpeak for text to phoneme conversion!",
                    lines=4,
                )
                language = gr.Dropdown(
                    choices=supported_language_codes,
                    value="ko",
                    label="Language Code",
                    info="Select a language code.",
                )
            prefix_audio = gr.Audio(
                value="assets/silence_100ms.wav",
                label="Optional Prefix Audio (continue from this audio)",
                type="filepath",
            )
            with gr.Column():
                speaker_audio = gr.Audio(
                    label="Optional Speaker Audio (for cloning)",
                    type="filepath",
                )
                speaker_noised_checkbox = gr.Checkbox(label="Denoise Speaker?", value=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Conditioning Parameters")
                dnsmos_slider = gr.Slider(1.0, 5.0, value=4.0, step=0.1, label="DNSMOS Overall")
                fmax_slider = gr.Slider(0, 24000, value=24000, step=1, label="Fmax (Hz)")
                vq_single_slider = gr.Slider(0.5, 0.8, 0.78, 0.01, label="VQ Score")
                pitch_std_slider = gr.Slider(0.0, 300.0, value=45.0, step=1, label="Pitch Std")
                speaking_rate_slider = gr.Slider(5.0, 30.0, value=15.0, step=0.5, label="Speaking Rate")

            with gr.Column():
                gr.Markdown("## Generation Parameters")
                cfg_scale_slider = gr.Slider(1.0, 5.0, 2.0, 0.1, label="CFG Scale")
                seed_number = gr.Number(label="Seed", value=420, precision=0)
                randomize_seed_toggle = gr.Checkbox(label="Randomize Seed (before generation)", value=True)

        with gr.Accordion("Sampling", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### NovelAi's unified sampler")
                    linear_slider = gr.Slider(-2.0, 2.0, 0.5, 0.01,
                                              label="Linear (set to 0 to disable unified sampling)",
                                              info="High values make the output less random.")
                    # Conf's theoretical range is between -2 * Quad and 0.
                    confidence_slider = gr.Slider(-2.0, 2.0, 0.40, 0.01, label="Confidence",
                                                  info="Low values make random outputs more random.")
                    quadratic_slider = gr.Slider(-2.0, 2.0, 0.00, 0.01, label="Quadratic",
                                                 info="High values make low probablities much lower.")
                with gr.Column():
                    gr.Markdown("### Legacy sampling")
                    top_p_slider = gr.Slider(0.0, 1.0, 0, 0.01, label="Top P")
                    min_k_slider = gr.Slider(0.0, 1024, 0, 1, label="Min K")
                    min_p_slider = gr.Slider(0.0, 1.0, 0, 0.01, label="Min P")

        with gr.Accordion("Advanced Parameters", open=False):
            gr.Markdown(
                "### Unconditional Toggles\n"
                "Checking a box will make the model ignore the corresponding conditioning value and make it unconditional.\n"
                'Practically this means the given conditioning feature will be unconstrained and "filled in automatically".'
            )
            with gr.Row():
                unconditional_keys = gr.CheckboxGroup(
                    [
                        "speaker",
                        "emotion",
                        "vqscore_8",
                        "fmax",
                        "pitch_std",
                        "speaking_rate",
                        "dnsmos_ovrl",
                        "speaker_noised",
                    ],
                    value=["emotion"],
                    label="Unconditional Keys",
                )

            gr.Markdown(
                "### Emotion Sliders\n"
                "Warning: The way these sliders work is not intuitive and may require some trial and error to get the desired effect.\n"
                "Certain configurations can cause the model to become unstable. Setting emotion to unconditional may help."
            )
            with gr.Row():
                emotion1 = gr.Slider(0.0, 1.0, 1.0, 0.05, label="Happiness")
                emotion2 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Sadness")
                emotion3 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Disgust")
                emotion4 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Fear")
            with gr.Row():
                emotion5 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Surprise")
                emotion6 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Anger")
                emotion7 = gr.Slider(0.0, 1.0, 0.1, 0.05, label="Other")
                emotion8 = gr.Slider(0.0, 1.0, 0.2, 0.05, label="Neutral")

        with gr.Column():
            generate_button = gr.Button("Generate Audio")
            output_audio = gr.Audio(label="Generated Audio", type="filepath", autoplay=True)

        model_choice.change(
            fn=update_ui,
            inputs=[model_choice],
            outputs=[
                text,
                language,
                speaker_audio,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                unconditional_keys,
            ],
        )

        # On page load, trigger the same UI refresh
        demo.load(
            fn=update_ui,
            inputs=[model_choice],
            outputs=[
                text,
                language,
                speaker_audio,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                unconditional_keys,
            ],
        )

        # Generate audio on button click
        generate_button.click(
            fn=generate_audio,
            inputs=[
                model_choice,
                text,
                language,
                speaker_audio,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                cfg_scale_slider,
                top_p_slider,
                min_k_slider,
                min_p_slider,
                linear_slider,
                confidence_slider,
                quadratic_slider,
                seed_number,
                randomize_seed_toggle,
                unconditional_keys,
            ],
            outputs=[output_audio, seed_number],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    share = getenv("GRADIO_SHARE", "False").lower() in ("true", "1", "t")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share)
