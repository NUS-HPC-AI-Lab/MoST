import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np
device = "cuda:0" if torch.cuda.is_available() else "cpu"
## print device info
print(f"Using device: {device}")

cache_dir = "path/to/data/ParlerTTS/"


model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-large-v1",cache_dir=cache_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-large-v1",cache_dir=cache_dir)

prompt = "Tell me about your favorite books and why you enjoy them."
description1 = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."
description2 = "Jenna's voice is monotone yet slightly slow in delivery, with a very close recording that almost has no background noise."
# prompt = "Hey, how are you doing today?"
# description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."

inputs = tokenizer(description1, return_tensors="pt").to(device)
prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)

generation = model.generate(input_ids=inputs.input_ids, prompt_input_ids=prompt_inputs.input_ids, 
                            attention_mask=inputs.attention_mask,
                            prompt_attention_mask=prompt_inputs.attention_mask)
audio_arr_1 = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out_1.wav", audio_arr_1, model.config.sampling_rate)

prompt2_1 = "As an assistant, I enjoy discussing various books across different genres. I find classic literature particularly fascinating because of its timeless themes and complex character development. For example, works like 'To Kill a Mockingbird' demonstrate powerful messages about justice and humanity."
prompt2_2 = "I'm also drawn to science fiction novels that explore the implications of technology and artificial intelligence on society. Books like 'Foundation' by Isaac Asimov"
inputs2 = tokenizer(description2, return_tensors="pt").to(device)
prompt_inputs2_1 = tokenizer(prompt2_1, return_tensors="pt").to(device)
prompt_inputs2_2 = tokenizer(prompt2_2, return_tensors="pt").to(device)

generation2_1 = model.generate(input_ids=inputs2.input_ids, prompt_input_ids=prompt_inputs2_1.input_ids, 
                            attention_mask=inputs2.attention_mask,
                            prompt_attention_mask=prompt_inputs2_1.attention_mask)
audio_arr_2_1 = generation2_1.cpu().numpy().squeeze()
sf.write("parler_tts_out_2_1.wav", audio_arr_2_1, model.config.sampling_rate)

generation2_2 = model.generate(input_ids=inputs2.input_ids, prompt_input_ids=prompt_inputs2_2.input_ids, 
                            attention_mask=inputs2.attention_mask,
                            prompt_attention_mask=prompt_inputs2_2.attention_mask)
audio_arr_2_2 = generation2_2.cpu().numpy().squeeze()
sf.write("parler_tts_out_2_2.wav", audio_arr_2_2, model.config.sampling_rate)


prompt3 = "Oh, I've read Foundation! But I found it pretty hard to follow. Did you think it was complicated?"
inputs3 = tokenizer(description1, return_tensors="pt").to(device)
prompt_inputs3 = tokenizer(prompt3, return_tensors="pt").to(device)

generation3 = model.generate(input_ids=inputs3.input_ids, prompt_input_ids=prompt_inputs3.input_ids, 
                            attention_mask=inputs3.attention_mask,
                            prompt_attention_mask=prompt_inputs3.attention_mask)
audio_arr_3 = generation3.cpu().numpy().squeeze()
sf.write("parler_tts_out_3.wav", audio_arr_3, model.config.sampling_rate)

prompt4 = "I'm glad you found it interesting! I think it's a great book, but I can see why it might be challenging for some readers. It's definitely worth a second read if you're interested in the topic."
inputs4 = tokenizer(description2, return_tensors="pt").to(device)
prompt_inputs4 = tokenizer(prompt4, return_tensors="pt").to(device)

generation4 = model.generate(input_ids=inputs4.input_ids, prompt_input_ids=prompt_inputs4.input_ids, 
                            attention_mask=inputs4.attention_mask,
                            prompt_attention_mask=prompt_inputs4.attention_mask)
audio_arr_4 = generation4.cpu().numpy().squeeze()
sf.write("parler_tts_out_4.wav", audio_arr_4, model.config.sampling_rate)

## concatenate all the audio files
audio_arr_all = np.concatenate([audio_arr_1, audio_arr_2_1, audio_arr_2_2, audio_arr_3, audio_arr_4], axis=0)
sf.write("parler_tts_out_all.wav", audio_arr_all, model.config.sampling_rate)

print('Done')
