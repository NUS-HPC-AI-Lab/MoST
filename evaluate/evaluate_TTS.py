import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import load_dataset
import jiwer
from tqdm import tqdm

def load_asr_model(model_id="facebook/hubert-large-ls960-ft"):
    """
    Load ASR model for transcribing generated speech
    Using HuBERT-Large finetuned on LibriSpeech as mentioned in the paper
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.to(device)
    return model, processor

def load_librispeech():
    """
    Load LibriSpeech test-clean dataset
    """
    dataset = load_dataset("librispeech_asr", "clean", split="test")
    return dataset

def generate_speech(text, tts_model):
    """
    Generate speech from text using your TTS model
    You'll need to implement this according to your TTS model's API
    """
    # This is a placeholder - implement according to your TTS model
    audio = tts_model.generate(text)
    return audio

def transcribe_audio(audio, asr_model, processor):
    """
    Transcribe audio using ASR model
    """
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).to(asr_model.device)
    
    with torch.no_grad():
        outputs = asr_model.generate(**inputs)
    
    transcription = processor.decode(outputs[0])
    return transcription

def compute_wer(hypotheses, references):
    """
    Compute Word Error Rate using jiwer
    """
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveWhitespace(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.SentencesToListOfWords(),
    ])
    
    return jiwer.compute_measures(
        references,
        hypotheses,
        truth_transform=transformation,
        hypothesis_transform=transformation
    )

def evaluate_tts(tts_model, asr_model, asr_processor, dataset, num_samples=None):
    """
    Evaluate TTS model using LibriSpeech test-clean
    """
    hypotheses = []
    references = []
    
    # Use subset of dataset if specified
    eval_data = dataset[:num_samples] if num_samples else dataset
    
    for example in tqdm(eval_data):
        try:
            # Generate speech from text
            generated_audio = generate_speech(example['text'], tts_model)
            
            # Transcribe generated speech
            transcription = transcribe_audio(generated_audio, asr_model, asr_processor)
            
            hypotheses.append(transcription)
            references.append(example['text'])
            
        except Exception as e:
            print(f"Error processing example: {str(e)}")
            continue
    
    # Compute WER
    results = compute_wer(hypotheses, references)
    
    return results, hypotheses, references

def main():
    # Load ASR model for transcription
    asr_model, asr_processor = load_asr_model()
    
    # Load your TTS model
    tts_model = None  # Implement loading your TTS model
    
    # Load dataset
    dataset = load_librispeech()
    
    # Run evaluation
    print("Starting TTS evaluation...")
    results, hypotheses, references = evaluate_tts(
        tts_model, 
        asr_model, 
        asr_processor, 
        dataset,
        num_samples=100  # Optional: evaluate on subset
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Word Error Rate: {results['wer']:.4f}")
    print(f"Match Error Rate: {results['mer']:.4f}")
    print(f"Word Information Loss: {results['wil']:.4f}")
    
    # Save detailed results
    with open("tts_evaluation_results.txt", "w") as f:
        f.write("Generated Speech Transcriptions vs References:\n\n")
        for hyp, ref in zip(hypotheses, references):
            f.write(f"Generated: {hyp}\nReference: {ref}\n\n")
        f.write("\nMetrics:\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")

if __name__ == "__main__":
    main()