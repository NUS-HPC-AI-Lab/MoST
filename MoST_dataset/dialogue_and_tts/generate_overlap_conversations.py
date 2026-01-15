import dataclasses
from typing import List, Tuple, Optional
from enum import Enum
import random

class SpeakerRole(Enum):
    MAIN = "MAIN"
    INTERRUPT = "INTERRUPT"
    BACKCHANNEL = "BACKCHANNEL"

@dataclasses.dataclass
class DialogueSegment:
    speaker: str
    text: str
    start_time: float  # in seconds
    duration: float    # in seconds
    role: SpeakerRole
    overlap_with: Optional[str] = None

class OverlappingDialogueGenerator:
    def __init__(self):
        self.average_word_duration = 0.3  # seconds per word
        self.backchannel_responses = [
            "mm-hmm", "yeah", "right", "okay", "I see",
            "uh-huh", "got it", "sure"
        ]
        
    def estimate_duration(self, text: str) -> float:
        """Estimate duration of speech segment based on word count."""
        return len(text.split()) * self.average_word_duration
    
    def generate_backchannel(self, main_segment: DialogueSegment, speaker: str) -> DialogueSegment:
        """Generate a backchannel response during the main speech."""
        response = random.choice(self.backchannel_responses)
        # Place backchannel randomly within the main segment
        start_offset = random.uniform(
            main_segment.duration * 0.3,  # Start after 30% of main segment
            main_segment.duration * 0.7    # End before 70% through
        )
        return DialogueSegment(
            speaker=speaker,
            text=response,
            start_time=main_segment.start_time + start_offset,
            duration=self.estimate_duration(response),
            role=SpeakerRole.BACKCHANNEL,
            overlap_with=main_segment.speaker
        )
    
    def generate_interruption(self, 
                            main_segment: DialogueSegment, 
                            speaker: str, 
                            interrupt_text: str) -> Tuple[DialogueSegment, DialogueSegment]:
        """Generate an interruption and modify the main segment."""
        # Cut the main segment short
        interrupt_point = random.uniform(
            main_segment.duration * 0.4,  # Don't interrupt too early
            main_segment.duration * 0.8   # Or too late
        )
        
        # Modified main segment
        truncated_main = dataclasses.replace(
            main_segment,
            duration=interrupt_point,
            text=main_segment.text + "—"  # Add cut-off marker
        )
        
        # Create interruption segment
        interruption = DialogueSegment(
            speaker=speaker,
            text=interrupt_text,
            start_time=main_segment.start_time + interrupt_point - 0.1,  # Slight overlap
            duration=self.estimate_duration(interrupt_text),
            role=SpeakerRole.INTERRUPT,
            overlap_with=main_segment.speaker
        )
        
        return truncated_main, interruption

def generate_conversation_example():
    generator = OverlappingDialogueGenerator()
    
    # Example 1: Backchannel during main speech
    main_speech = DialogueSegment(
        speaker="Speaker1",
        text="I was thinking about what you said yesterday regarding the project timeline, and I believe we might need to adjust our expectations.",
        start_time=0.0,
        duration=5.0,
        role=SpeakerRole.MAIN
    )
    
    backchannel = generator.generate_backchannel(main_speech, "Speaker2")
    
    # Example 2: Interruption
    main_speech2 = DialogueSegment(
        speaker="Speaker1",
        text="The problem with this approach is that it doesn't take into account the long-term implications of",
        start_time=6.0,
        duration=4.0,
        role=SpeakerRole.MAIN
    )
    
    truncated_main, interruption = generator.generate_interruption(
        main_speech2,
        "Speaker2",
        "But we've already discussed this in the previous meeting!"
    )
    
    return [main_speech, backchannel, truncated_main, interruption]

# Generate and display example conversation
# conversation = generate_conversation_example()
# for segment in conversation:
#     print(f"\nTime: {segment.start_time:.1f}s - {segment.start_time + segment.duration:.1f}s")
#     print(f"{segment.speaker} ({segment.role.value}): {segment.text}")
#     if segment.overlap_with:
#         print(f"Overlaps with: {segment.overlap_with}")

prompt1 = "You are tasked with determining if a dialogue is suitable for adding interruptions. A suitable dialogue should have: \n \
- At least one longer turn (where interruption would be natural)\n\
- Content that could trigger reactions/interruptions (opinions, explanations, or detailed information)\n\
- Natural flow between speakers \n\
\n\
Analyze the following dialogue and respond with:\n\
1. YES/NO decision \n\
2. Brief explanation of why \n\
3. If YES, identify specific turns that are good candidates for interruption \n\
\n\
Dialogue:\n\
\
"
prompt2 = "You are tasked with modifying a dialogue to include natural interruptions. Follow these rules: \n\
1. When a speaker is interrupted, they should stop mid-sentence \n\
2. The interrupting speaker may add a brief acknowledgment of interrupting \n\
3. Include natural breaking points and partial words where appropriate \n\
4. The interrupted speaker might briefly acknowledge being interrupted \n\
\n\
Original dialogue: \n\
 \n\
\n\
Please: \n\
1. Choose ONE natural interruption point \n\
2. Modify the dialogue to include the interruption \n\
3. Format the output as: \n\
Speaker A: [Original start of utterance-] \n\
Speaker B: [Interruption] \n\
Speaker A: [Optional acknowledgment] \n\
[Continue dialogue] \n\
"
print(prompt2)