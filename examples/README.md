# MoST Examples

This directory contains example scripts for using the MoST (Mixture of Speech and Text) model, particularly with the Modality-Aware Mixture of Experts (MAMoE) routing mechanism.

## Example Scripts

### 1. MAMoE Usage Example
File: `mamoe_usage_example.py`

This script demonstrates how to:
- Initialize a MoST model with MAMoE routing enabled
- Configure text and audio expert assignments
- Create and process mixed modality inputs (text and audio tokens)
- Run inference with the model

Usage:
```bash
python mamoe_usage_example.py
```

### 2. MAMoE Routing Validation
File: `validate_mamoe_routing.py`

This script provides a way to validate that the MAMoE routing mechanism is functioning correctly by:
- Setting up hooks to capture routing decisions
- Analyzing expert assignments based on token modality
- Calculating statistics about routing behavior
- Comparing standard MoE routing with MAMoE routing

Usage:
```bash
python validate_mamoe_routing.py
```

## Key MAMoE Configuration Parameters

When using MAMoE, you'll need to set these key configuration parameters:

```python
config = MoSTConfig(
    # Enable modality-aware routing
    use_modality_aware_routing=True,
    
    # Specify which experts handle text tokens
    text_expert_indices=[0, 1, 2, 3],
    
    # Specify which experts handle audio tokens
    audio_expert_indices=[4, 5, 6, 7],
    
    # Other standard MoE parameters
    n_shared_experts=1,  # Experts accessible to all modalities
    n_routed_experts=8,  # Total number of routed experts
    num_experts_per_tok=2,  # Number of experts to select per token
)
```

## Token Modality Determination

The model automatically determines token modality based on token ID ranges:
- 0-100001: Text tokens (modality mark = 0)
- 100002-100503: Audio tokens (modality mark = 1)
- 100503-102400: Image tokens (modality mark = 2)

For detailed implementation, see the `_get_modality_marks` method in `MoSTForCausalLM`. 