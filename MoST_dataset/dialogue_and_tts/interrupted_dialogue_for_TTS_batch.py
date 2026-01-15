import os
import json
from typing import List, Dict, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

class DialogueProcessor:
    def __init__(self, model_name: str = "Qwen/Qwen1.5-7B", model_cache_dir: str = "Qwen/Qwen1.5-7B"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=model_cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=model_cache_dir, padding_side='left')
        
        self.model.eval()  # Set eval
        
        # Define prompts
        self.suitability_prompt = """You are tasked with determining if a user-assistant dialogue is suitable for adding user interruptions. 

Rules:
- Only consider dialogues with clear user/assistant roles
- Only the user can interrupt the assistant
- The assistant must have at least one longer response that could be interrupted
- The interruption should feel natural given the context

Analyze the following dialogue and respond with:
1. YES/NO decision
2. Brief explanation of why

Dialogue:
{dialogue}
"""
    
        self.interruption_prompt = """Transform the following dialogue by adding ONE natural interruption. The interruption should occur during an assistant's explanation, and the assistant should briefly address it before continuing.

Example:

Original:
User: What's the difference between a virus and bacteria?
Assistant: Viruses and bacteria are very different types of microorganisms. Bacteria are living cells that can reproduce on their own, while viruses need a host cell to multiply.
User: How do we treat them?
Assistant: The treatment approaches are quite different. Bacterial infections can be treated with antibiotics, which kill the bacteria or stop them from reproducing. Viral infections, however, don't respond to antibiotics and usually require antiviral medications or just time for your immune system to fight them off.

Transformed:
User: What's the difference between a virus and bacteria?
Assistant: Viruses and bacteria are very different types of microorganisms. Bacteria are living cells that can reproduce on their own, while viruses need a host cell to...
User: Sorry, what do you mean by host cell?
Assistant: A host cell is a living cell that a virus infects and uses to make copies of itself. Now, as I was saying, viruses need a host cell to multiply.
User: How do we treat them?
Assistant: The treatment approaches are quite different. Bacterial infections can be treated with antibiotics, which kill the bacteria or stop them from reproducing. Viral infections, however, don't respond to antibiotics and usually require antiviral medications or just time for your immune system to fight them off.

Transform this dialogue:
{dialogue}

Rules:
1. Start with the complete original dialogue
2. Choose ONE natural point during an assistant's longer explanation to add the interruption
3. Split the assistant's response at the interruption point
4. Add a brief user question about what was just mentioned
5. Have the assistant briefly answer the question, then continue with "Now, as I was saying" or similar transition
6. Complete the rest of the original dialogue unchanged, including any subsequent turns

DO NOT:
- Include meta-instructions or [bracketed text]
- Repeat or provide multiple versions
- Include the prompt or requirements in the output
- Start from the middle of the dialogue
- Change any parts of the dialogue except for the interrupted section

Output the complete transformed dialogue in User/Assistant format only."""

    def format_dialogue(self, conversation: List[Dict]) -> str:
        """Format conversation into a string with clear speaker labels."""
        formatted = []
        for turn in conversation:
            role = "User" if turn["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {turn['content']}")
        return "\n".join(formatted)

    def format_back_dialogue(self, dialogue: str) -> List[Dict]:
        """Format back dialogue into a list of dictionaries with clear speaker labels."""
        formatted = []
        
        # Find the first occurrence of either "Assistant:" or "User:"
        dialogue_lines = dialogue.split('\n')
        start_idx = 0
        for idx, line in enumerate(dialogue_lines):
            if line.strip().startswith(("Assistant:", "User:")):
                start_idx = idx
                break
        
        # Remove any text before the actual dialogue starts
        dialogue_lines = dialogue_lines[start_idx:]
        
        # Rest of the dialogue processing
        turns = []
        current_turn = ""
        for line in dialogue_lines:
            if line.startswith("Assistant:") or line.startswith("User:"):
                if current_turn:
                    turns.append(current_turn.strip())
                current_turn = line
            else:
                current_turn += " " + line.strip()
        if current_turn:
            turns.append(current_turn.strip())
        
        # Convert turns into dictionaries
        for turn in turns:
            if turn.startswith("User:"):
                role = "user"
                content = turn[len("User:"):].strip()
            else:  # Assistant turn
                role = "assistant"
                content = turn[len("Assistant:"):].strip()
            formatted.append({"role": role, "content": content})
        
        return formatted

    def check_suitability_batch(self, dialogues: List[str]) -> List[Tuple[bool, str]]:
        """Check if dialogue is suitable for interruption."""
        prompts = [self.suitability_prompt.format(dialogue=dialogue) for dialogue in dialogues]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True
            )
        
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        results = []
        for prompt, response in zip(prompts, responses):
            # remove input from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            # Extract YES/NO decision and explanation from response
            first_line = response.split("\n")[0].upper()
            is_suitable = "YES" in first_line
            results.append((is_suitable, response))
        
        return results

    def add_interruption_batch(self, dialogues: List[str]) -> List[str]:
        """Add interruption to suitable dialogue."""
        prompts = [self.interruption_prompt.format(dialogue=dialogue) for dialogue in dialogues]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True
            )
        
        modified_dialogues = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # remove input from response
        modified_dialogues = [modified_dialogue[len(prompt):].strip() 
                              for prompt, modified_dialogue in zip(prompts, modified_dialogues)]
        
        return modified_dialogues

def main():
    batch_size = 2  # Set batch

    # Initialize model
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_cache_dir = "path/to/model"
    dataset_cache_dir = "path/to/smoltalk"
    dataset_name = "HuggingFaceTB/smoltalk"

    # Load data
    dataset = load_dataset(dataset_name, 'all', split="train[:1000]", cache_dir=dataset_cache_dir)
    print('Dataset loaded')

    # Load model
    processor = DialogueProcessor(model_name=model_name, model_cache_dir=model_cache_dir)
    print('Processor initialized')

    # Process dialogue
    processed_dialogues = []
    num_interrupted = 0
    dataset = list(dataset) # To support batch process

    # Batch process
    for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        batch = dataset[batch_start:batch_start + batch_size]
        indices = range(batch_start, batch_start + len(batch))

        # Format dialogue
        formatted_dialogues = [processor.format_dialogue(conversation["messages"]) for conversation in batch]
        
        # Check suitability
        suitability_results = processor.check_suitability_batch(formatted_dialogues)

        suitable_dialogues = []
        suitable_indices = []
        suitable_explanations = []
        for idx, (is_suitable, explanation) in zip(indices, suitability_results):
            # log
            with open("interrupt_log.txt", "a") as f:
                f.write(f"Idx: {idx}, Is_suitable: {is_suitable}\n")
            if is_suitable:
                suitable_dialogues.append(formatted_dialogues[idx - batch_start])
                suitable_indices.append(idx)
                suitable_explanations.append(explanation)
        
        if suitable_dialogues:
            # Add interruption
            modified_dialogues = processor.add_interruption_batch(suitable_dialogues)
            
            formatted_back_dialogues = [processor.format_back_dialogue(modified_dialogue) 
                                        for modified_dialogue in modified_dialogues]
            
            for idx, original, modified_unformatted, modified, explanation in zip(
                suitable_indices, suitable_dialogues, modified_dialogues, formatted_back_dialogues, suitable_explanations
            ):
                processed_dialogues.append({
                    "idx": idx,
                    "original": original,
                    "modified_unformatted": modified_unformatted,
                    "modified": modified,
                    "explanation": explanation
                })
                num_interrupted += 1

    print("Number of interrupted dialogues:", num_interrupted)

    # Save processed data
    output_dir = "path/to/processed_dialogues"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "interrupted_dialogues.json"), "w") as f:
        json.dump(processed_dialogues, f, indent=2)

if __name__ == "__main__":
    main()