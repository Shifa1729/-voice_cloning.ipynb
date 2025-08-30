# -voice_cloning.ipynb
# Cell 1: Installation and Setup
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes xformers datasets torchaudio soundfile librosa noisereduce

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, Audio
from trl import SFTTrainer
from transformers import TrainingArguments
import librosa
import noisereduce as nr
import soundfile as sf
import torchaudio
import numpy as np

# Cell 2: Data Preparation (Clean and Load Dataset)
# Load Elise dataset, slice to ~30 min (200 samples) for low-data test
dataset = load_dataset("MrDragonFox/Elise", split="train")
dataset = dataset.select(range(200))  # ~30 min

# Resample audio to 24kHz
dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))
# Cleaning function (high-pass filter, denoise, normalize)
def clean_audio(example):
    audio = example["audio"]["array"]
    sr = 24000
    # High-pass filter
    audio = librosa.effects.preemphasis(audio, coef=0.97)
    # Denoise
    audio = nr.reduce_noise(y=audio, sr=sr)
    # Normalize
    audio = librosa.util.normalize(audio)
    # Add 100ms silence
    audio = np.append(audio, np.zeros(int(0.1 * sr)))
    example["audio"]["array"] = audio
    return example

dataset = dataset.map(clean_audio)
# Add speaker ID (single speaker = 0)
dataset = dataset.add_column("speaker_id", [0] * len(dataset))

# Note: Transcription skipped since Elise dataset includes 'text' field
# If needed, you can manually add transcriptions or use another tool

# Cell 3: Load Model and Apply LoRA
max_seq_length = 2048
dtype = None  # Auto (bfloat16 if supported)
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/csm-1b",  # Sesame CSM-1B
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
# Cell 4: Training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",  # Use provided transcriptions
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,  # ~1 hour on T4
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

trainer.train()

# Save model
model.save_pretrained("fine_tuned_sesame")
tokenizer.save_pretrained("fine_tuned_sesame")
Cell 5: Inference (Generate Cloned Audio)
FastLanguageModel.for_inference(model)

# Use reference audio for voice cloning context
context_audio = dataset[0]["audio"]["array"]  # First sample as reference

# Generate from text
text = "Hello, this is a test of the cloned voice speaking a thirty second sample sentence to demonstrate the quality."
inputs = tokenizer(text, return_tensors="pt").to("cuda")

# Generate audio (Sesame outputs tokens; simplified waveform handling)
outputs = model.generate(**inputs, max_new_tokens=512)  # Adjust for length

# Convert outputs to waveform (placeholder; Sesame may require vocoder)
# Note: If waveform output is distorted, integrate a vocoder like HiFi-GAN
audio_waveform = outputs[0].cpu().float()  # Assuming model outputs float waveform

Save cloned audio
torchaudio.save("cloned.wav", audio_waveform.unsqueeze(0), sample_rate=24000)

# Save original sample for comparison (30s clip)
original_audio = dataset[0]["audio"]["array"][:24000*30]  # First 30s
sf.write("original.wav", original_audio, 24000)

print("Audio samples saved: original.wav and cloned.wav")
