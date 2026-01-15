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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=model_cache_dir)
        
        
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
        

    def check_suitability(self, dialogue: str) -> Tuple[bool, str]:
        """Check if dialogue is suitable for interruption."""
        prompt = self.suitability_prompt.format(dialogue=dialogue)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # remove input from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        # Extract YES/NO decision and explanation from response
        is_suitable = "YES" in response.split("\n")[0].upper()
        return is_suitable, response

    def add_interruption(self, dialogue: str) -> str:
        """Add interruption to suitable dialogue."""
        prompt = self.interruption_prompt.format(dialogue=dialogue)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True
        )
        
        modified_dialogue = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # remove input from response
        modified_dialogue = modified_dialogue[len(prompt):].strip()
        return modified_dialogue

def main():
    # Initialize processor
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_cache_dir = "path/to/scratch/model"
    processor = DialogueProcessor(model_name=model_name, model_cache_dir=model_cache_dir)

    print('processor initialized')

    dataset_cache_dir = "path/to/smoltalk"
    dataset_name = "HuggingFaceTB/smoltalk"
    # Load SmolTalk dataset
    # dataset = load_dataset(dataset_name, 'all',split="train", cache_dir=dataset_cache_dir)
    dataset = load_dataset(dataset_name, 'all', split="train[:1000]", cache_dir=dataset_cache_dir)
    
    print('dataset loaded')
    
    # Process dialogues
    processed_dialogues = []
    
    num_interrupted = 0
    for idx, conversation in tqdm(enumerate(dataset)):
        # Format dialogue
        formatted_dialogue = processor.format_dialogue(conversation["messages"])
        
        # Check suitability
        is_suitable, explanation = processor.check_suitability(formatted_dialogue)

        with open("interrupt_log.txt", "a") as f:
            f.write(f"Idx: {idx}, Is_suitable: {is_suitable}\n")

        if not is_suitable:
            continue
        
        if is_suitable:
            # Add interruption
            modified_dialogue = processor.add_interruption(formatted_dialogue)
            num_interrupted += 1
            formatted_back_dialogue = processor.format_back_dialogue(modified_dialogue)
            processed_dialogues.append({
                "idx": idx,
                "original": formatted_dialogue,
                "modified_unformatted": modified_dialogue,
                "modified": formatted_back_dialogue,
                "explanation": explanation
            })

            
    print("num_interrupted: ", num_interrupted)
    # Save processed dialogues
    output_dir = "path/to/smoltalk/processed_dialogues"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "interrupted_dialogues.json"), "w") as f:
        json.dump(processed_dialogues, f, indent=2)

if __name__ == "__main__":
    main()
