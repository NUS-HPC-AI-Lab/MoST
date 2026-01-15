import torch
import torchaudio
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import editdistance

def load_model_and_processor(model_id="kyutai/moshiko-pytorch-bf16"):
    """Load the model and processor."""
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

def load_webquestions_dataset():
    """Load the WebQuestions dataset."""
    dataset = load_dataset("web_questions")
    return dataset["test"]  # Using test split for evaluation

def preprocess_audio(audio_path, processor):
    """Load and preprocess audio for the model."""
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    inputs = processor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    )
    return inputs

def normalize_answer(s):
    """Normalize answer string for comparison."""
    return " ".join(s.lower().split())

def compute_metrics(predictions, references):
    """Compute evaluation metrics."""
    exact_matches = 0
    normalized_edit_distances = []
    
    for pred, ref in zip(predictions, references):
        pred = normalize_answer(pred)
        ref = normalize_answer(ref)
        
        # Compute exact match
        if pred == ref:
            exact_matches += 1
            
        # Compute normalized edit distance
        max_len = max(len(pred), len(ref))
        if max_len > 0:
            normalized_edit_distances.append(
                editdistance.eval(pred, ref) / max_len
            )
    
    metrics = {
        "exact_match": 100 * exact_matches / len(predictions),
        "avg_edit_distance": np.mean(normalized_edit_distances),
    }
    return metrics

def evaluate_spoken_qa(model, processor, dataset, audio_dir):
    """Evaluate the model on spoken question answering."""
    predictions = []
    references = []
    
    for example in tqdm(dataset):
        # Construct audio path - modify according to your directory structure
        audio_path = f"{audio_dir}/{example['id']}.wav"
        
        try:
            # Process audio and generate transcription
            inputs = preprocess_audio(audio_path, processor)
            with torch.no_grad():
                outputs = model.generate(**inputs)
            transcription = processor.decode(outputs[0])
            
            predictions.append(transcription)
            references.append(example['answers'][0])  # Using first answer as reference
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            continue
    
    # Compute metrics
    metrics = compute_metrics(predictions, references)
    return metrics, predictions, references

def main():
    # Initialize model and processor
    model, processor = load_model_and_processor()
    
    # Load dataset
    dataset = load_webquestions_dataset()
    
    # Set your audio directory path
    audio_dir = "path/to/audio/files"
    
    # Run evaluation
    metrics, predictions, references = evaluate_spoken_qa(
        model, processor, dataset, audio_dir
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Exact Match: {metrics['exact_match']:.2f}%")
    print(f"Average Edit Distance: {metrics['avg_edit_distance']:.4f}")
    
    # Save detailed results
    with open("evaluation_results.txt", "w") as f:
        f.write("Predictions vs References:\n\n")
        for pred, ref in zip(predictions, references):
            f.write(f"Pred: {pred}\nRef:  {ref}\n\n")
        f.write("\nMetrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")

if __name__ == "__main__":
    main()