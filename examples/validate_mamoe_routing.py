#!/usr/bin/env python
# coding=utf-8

"""
Validation script to verify the correctness of Modality-Aware Mixture of Experts (MAMoE) routing
in the MoST (Mixture of Speech and Text) model.
"""

import torch
import torch.nn.functional as F
from configuration_MoST import MoSTConfig
from modeling_most import MoSTForCausalLM, MAMoEGate, MoEGate

class RoutingValidator:
    """Helper class to validate MAMoE routing behavior"""
    
    def __init__(self, model):
        self.model = model
        self.config = model.config
        
    def extract_gate_decisions(self, input_ids, attention_mask=None):
        """
        Extract and analyze routing decisions from the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary containing routing statistics
        """
        # Set up hooks to capture gate decisions
        gate_decisions = {}
        
        def gate_hook(name):
            def hook_fn(module, inputs, outputs):
                # For MAMoEGate, inputs will contain modality_marks
                if isinstance(module, MAMoEGate):
                    # inputs[0] is hidden_states, inputs[1] is modality_marks
                    hidden_states, modality_marks = inputs
                    token_modalities = modality_marks.view(-1)
                    
                    # outputs[0] is topk_idx, outputs[1] is topk_weight
                    expert_indices, expert_weights, _ = outputs
                    
                    # Store for analysis
                    gate_decisions[name] = {
                        'expert_indices': expert_indices.detach().cpu(),
                        'expert_weights': expert_weights.detach().cpu(),
                        'token_modalities': token_modalities.detach().cpu(),
                        'hidden_states': hidden_states.detach().cpu()
                    }
                # For regular MoEGate
                elif isinstance(module, MoEGate) and not isinstance(module, MAMoEGate):
                    # inputs[0] is hidden_states
                    hidden_states = inputs[0]
                    
                    # outputs[0] is topk_idx, outputs[1] is topk_weight
                    expert_indices, expert_weights, _ = outputs
                    
                    # Store for analysis
                    gate_decisions[name] = {
                        'expert_indices': expert_indices.detach().cpu(),
                        'expert_weights': expert_weights.detach().cpu(),
                        'hidden_states': hidden_states.detach().cpu()
                    }
            return hook_fn
        
        # Register hooks on all MAMoE gates
        hooks = []
        layer_idx = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, MAMoEGate):
                hooks.append((name, module.register_forward_hook(gate_hook(f"layer_{layer_idx}"))))
                layer_idx += 1
        
        # Get modality marks
        modality_marks = torch.zeros_like(input_ids)
        modality_marks[(input_ids >= 100002) & (input_ids <= 100503)] = 1  # Audio tokens
        modality_marks[input_ids > 100503] = 2  # Image tokens
        
        # Run forward pass to collect gate decisions
        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask, modality_marks=modality_marks)
        
        # Remove hooks
        for _, hook in hooks:
            hook.remove()
        
        # Analyze routing decisions
        stats = self._analyze_routing(gate_decisions, modality_marks)
        
        return stats
    
    def _analyze_routing(self, gate_decisions, modality_marks):
        """Analyze the collected gate decisions"""
        stats = {}
        
        for gate_name, decisions in gate_decisions.items():
            expert_indices = decisions['expert_indices']
            token_modalities = decisions.get('token_modalities')
            
            if token_modalities is None:
                continue
                
            # Flatten batch and sequence dimensions
            expert_indices = expert_indices.view(-1, expert_indices.size(-1))
            token_modalities = token_modalities.view(-1)
            
            # Separate text and audio tokens
            text_mask = token_modalities == 0
            audio_mask = token_modalities == 1
            
            text_expert_indices = expert_indices[text_mask].view(-1)
            audio_expert_indices = expert_indices[audio_mask].view(-1)
            
            # Check if text tokens only go to text experts
            text_experts_set = set(self.config.text_expert_indices)
            audio_experts_set = set(self.config.audio_expert_indices)
            
            # Analyze text tokens
            text_to_text_experts = sum(idx.item() in text_experts_set for idx in text_expert_indices)
            text_to_audio_experts = sum(idx.item() in audio_experts_set for idx in text_expert_indices)
            
            # Analyze audio tokens
            audio_to_text_experts = sum(idx.item() in text_experts_set for idx in audio_expert_indices)
            audio_to_audio_experts = sum(idx.item() in audio_experts_set for idx in audio_expert_indices)
            
            # Calculate percentages
            total_text_decisions = len(text_expert_indices) if len(text_expert_indices) > 0 else 1
            total_audio_decisions = len(audio_expert_indices) if len(audio_expert_indices) > 0 else 1
            
            text_to_text_pct = 100 * text_to_text_experts / total_text_decisions
            text_to_audio_pct = 100 * text_to_audio_experts / total_text_decisions
            audio_to_text_pct = 100 * audio_to_text_experts / total_audio_decisions
            audio_to_audio_pct = 100 * audio_to_audio_experts / total_audio_decisions
            
            stats[gate_name] = {
                'text': {
                    'to_text_experts': text_to_text_pct,
                    'to_audio_experts': text_to_audio_pct,
                    'total_decisions': total_text_decisions
                },
                'audio': {
                    'to_text_experts': audio_to_text_pct,
                    'to_audio_experts': audio_to_audio_pct,
                    'total_decisions': total_audio_decisions
                }
            }
        
        return stats

def main():
    # Initialize model with MAMoE routing
    config = MoSTConfig(
        # Model architecture parameters (smaller for testing)
        vocab_size=102400,
        hidden_size=768,
        num_hidden_layers=4,
        num_attention_heads=12,
        
        # MoE parameters
        n_shared_experts=1,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_layer_freq=2,  # Every 2nd layer is MoE
        
        # MAMoE specific parameters
        use_modality_aware_routing=True,
        text_expert_indices=[0, 1, 2, 3],  # First 4 experts for text
        audio_expert_indices=[4, 5, 6, 7],  # Last 4 experts for audio
    )
    
    print("Initializing model with MAMoE routing...")
    model = MoSTForCausalLM(config)
    
    # Create mixed input with both text and audio tokens
    batch_size = 2
    seq_len = 10
    
    # Generate mixed input (5 text tokens, 5 audio tokens)
    input_ids = torch.randint(0, 100001, (batch_size, seq_len//2))  # Text tokens
    audio_ids = torch.randint(100002, 100503, (batch_size, seq_len//2))  # Audio tokens
    
    # Concatenate to create mixed input
    input_ids = torch.cat([input_ids, audio_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    
    # Create validator
    validator = RoutingValidator(model)
    
    print("\nValidating MAMoE routing...")
    stats = validator.extract_gate_decisions(input_ids, attention_mask)
    
    # Print results
    print("\nRouting Statistics:")
    print("==================")
    for gate_name, gate_stats in stats.items():
        print(f"\n{gate_name}:")
        
        print("  Text tokens:")
        print(f"    - Routed to text experts: {gate_stats['text']['to_text_experts']:.2f}%")
        print(f"    - Routed to audio experts: {gate_stats['text']['to_audio_experts']:.2f}%")
        
        print("  Audio tokens:")
        print(f"    - Routed to text experts: {gate_stats['audio']['to_text_experts']:.2f}%")
        print(f"    - Routed to audio experts: {gate_stats['audio']['to_audio_experts']:.2f}%")
    
    # Validate correctness
    is_valid = True
    for gate_stats in stats.values():
        if gate_stats['text']['to_audio_experts'] > 0 or gate_stats['audio']['to_text_experts'] > 0:
            is_valid = False
            break
    
    if is_valid:
        print("\n✅ MAMoE routing is working correctly!")
        print("- Text tokens are only routed to text experts")
        print("- Audio tokens are only routed to audio experts")
    else:
        print("\n❌ MAMoE routing has issues!")
        print("Some tokens are being routed to the wrong modality experts.")
    
    # Compare with standard MoE routing
    print("\nComparing with standard MoE routing...")
    config.use_modality_aware_routing = False
    standard_model = MoSTForCausalLM(config)
    
    # Run the same validation
    validator = RoutingValidator(standard_model)
    
    # This will collect routing information, but won't validate MAMoE behavior
    # since the model is using standard routing
    
    print("\nStandard MoE routing doesn't separate experts by modality.")
    print("Both text and audio tokens are routed to all experts based only on content.")

if __name__ == "__main__":
    main() 