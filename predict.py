# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import re
import time
import torch
import subprocess
from cog import BasePredictor, Input
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set HuggingFace to offline mode to use local files only
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

MODEL_PATH = "./checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/openai/gpt-oss-safeguard-20b/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download weights
        if not os.path.exists(MODEL_PATH):
            download_weights(MODEL_URL, MODEL_PATH)

        print(f"Loading model from {MODEL_PATH}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()

        print("Model loaded successfully!")

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt or question to evaluate",
            default="Your bank details are needed to complete this transaction."
        ),
        policy: str = Input(
            description="Safety policy or rules to evaluate the prompt against. This defines the criteria for classification.",
            default=""
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate",
            default=512,
            ge=1,
            le=8192
        ),
        temperature: float = Input(
            description="Sampling temperature (0 for greedy decoding)",
            default=1.0,
            ge=0.0,
            le=2.0
        ),
        top_p: float = Input(
            description="Nucleus sampling top-p",
            default=1.0,
            ge=0.0,
            le=1.0
        ),
        repetition_penalty: float = Input(
            description="Repetition penalty to reduce repetitive text (1.0 = no penalty)",
            default=1.0,
            ge=0.8,
            le=2.0
        ),
        reasoning_effort: str = Input(
            description="Reasoning effort level: 'low' for faster responses on straightforward cases, 'medium' for balanced reasoning (default), or 'high' for deeper analysis of complex edge cases",
            default="medium",
            choices=["low", "medium", "high"]
        ),
    ) -> dict:
        """Run a single prediction on the model"""
        
        # Format messages using harmony format
        # Policy goes in developer role (instructions), prompt goes in user role
        messages = []
        
        if policy:
            messages.append({
                "role": "developer",
                "content": policy
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Apply chat template with reasoning_effort parameter
        # The harmony format uses reasoning_effort to control reasoning depth
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=reasoning_effort
        )
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=131072,  # Max context length is 131K
        ).to(self.device)

        # Determine sampling strategy
        do_sample = temperature > 0.0

        # Prepare generation kwargs
        # EOS token IDs from generation_config: [200002, 199999]
        # 200002 = <|return|>, 199999 = <|endoftext|>
        eos_token_ids = [200002, 199999]
        
        generate_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": eos_token_ids,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        # Add sampling parameters if using sampling
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
            if repetition_penalty != 1.0:
                generate_kwargs["repetition_penalty"] = repetition_penalty

        # Generate output
        with torch.inference_mode():
            generated_ids = self.model.generate(**generate_kwargs)

        # Decode output (don't skip special tokens initially to preserve format markers)
        output_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )

        # Remove the input prompt from the output if it's included
        if formatted_prompt in output_text:
            output_text = output_text.split(formatted_prompt, 1)[-1].strip()

        # Extract analysis and answer from harmony format
        # The model outputs multiple possible formats:
        # Format 1: <|start|>assistant<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...<|return|>
        # Format 2: Raw text with analysis content mixed in (fallback)
        analysis = ""
        answer = ""
        
        # Debug: Print raw output preview
        print(f"DEBUG: Raw output preview (first 1000 chars):\n{output_text[:1000]}\n")
        
        # Pattern 1: Try proper harmony format markers first
        # Analysis channel: <|start|>assistant<|channel|>analysis<|message|>content<|end|>
        analysis_patterns = [
            r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>',
            r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>',
            r'assistant<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>',
        ]
        
        for pattern in analysis_patterns:
            analysis_match = re.search(pattern, output_text, re.DOTALL)
            if analysis_match:
                analysis = analysis_match.group(1).strip()
                break
        
        # Final channel: <|start|>assistant<|channel|>final<|message|>content<|return|> or <|end|>
        final_patterns = [
            r'<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)(?:<\|return\||<\|end\||$)',
            r'<\|channel\|>final<\|message\|>(.*?)(?:<\|return\||<\|end\||$)',
            r'assistant<\|channel\|>final<\|message\|>(.*?)(?:<\|return\||<\|end\||$)',
        ]
        
        for pattern in final_patterns:
            final_match = re.search(pattern, output_text, re.DOTALL)
            if final_match:
                answer = final_match.group(1).strip()
                break
        
        # Pattern 2: If no proper markers found, try to split on final channel marker
        if not answer and '<|channel|>final' in output_text:
            parts = output_text.split('<|channel|>final', 1)
            if len(parts) == 2:
                potential_analysis = parts[0]
                potential_answer = parts[1]
                # Extract content after <|message|> if present
                if '<|message|>' in potential_answer:
                    potential_answer = potential_answer.split('<|message|>', 1)[-1]
                # Clean up answer end markers
                potential_answer = re.sub(r'<\|return\||<\|end\||$', '', potential_answer, flags=re.DOTALL).strip()
                if potential_answer:
                    answer = potential_answer
                # Extract analysis if present
                if '<|channel|>analysis' in potential_analysis:
                    analysis_match = re.search(r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>', potential_analysis, re.DOTALL)
                    if analysis_match:
                        analysis = analysis_match.group(1).strip()
        
        # Pattern 3: Handle case where output starts with "analysis" (text, not marker)
        # This handles outputs like "analysisThe user says..." where analysis is part of content
        if not analysis and not answer:
            # Check if output starts with reasoning-like content
            if output_text.lower().startswith('analysis') or 'analysis' in output_text[:100].lower():
                # Try to find a natural split point - look for markers or transitions
                # Option A: Look for <|channel|>final or similar markers
                if '<|channel|>final' in output_text or '<|start|>assistant<|channel|>final' in output_text:
                    # Split on final marker
                    parts = re.split(r'<\|.*?channel.*?final.*?message.*?\|>', output_text, flags=re.IGNORECASE | re.DOTALL)
                    if len(parts) >= 2:
                        analysis = parts[0].strip()
                        answer = parts[-1].strip()
                # Option B: Look for transition phrases that suggest final answer
                else:
                    # Try to find where reasoning ends and answer begins
                    # Common patterns: "Thus", "Therefore", "Conclusion", or double newline
                    transition_patterns = [
                        r'(.*?)(?:\n\n|Thus|Therefore|Conclusion|Final answer|Answer:)(.*)',
                        r'(.*?)(<\|.*?final.*?\|>)(.*)',
                    ]
                    for pattern in transition_patterns:
                        match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)
                        if match and len(match.groups()) >= 2:
                            potential_analysis = match.group(1).strip()
                            potential_answer = match.group(-1).strip()  # Last group
                            if len(potential_answer) > 20:  # Ensure answer has content
                                analysis = potential_analysis
                                answer = potential_answer
                                break
        
        # Pattern 4: Ultimate fallback - output doesn't have proper format
        if not analysis and not answer:
            # Check if it looks like reasoning content (long, explains process)
            if len(output_text) > 200:
                # Try to find where reasoning might end and answer begins
                # Look for sentences that sound like conclusions
                sentences = re.split(r'[.!?]\s+', output_text)
                if len(sentences) > 3:
                    # First 60% as analysis, last 40% as answer
                    split_point = int(len(sentences) * 0.6)
                    analysis = '. '.join(sentences[:split_point]) + '.'
                    answer = '. '.join(sentences[split_point:]) + '.'
                else:
                    # Everything as answer if can't split meaningfully
                    answer = output_text.strip()
            else:
                # Short output - everything as answer
                answer = output_text.strip()
        
        # Clean up: Remove special tokens and markers
        def clean_text(text):
            if not text:
                return ""
            # Remove harmony format markers
            text = re.sub(r'<\|[^|]+\|>', '', text)
            # Remove leading "analysis" if it's just text (not a marker)
            text = re.sub(r'^\s*analysis\s*:?\s*', '', text, flags=re.IGNORECASE)
            # Clean up excessive whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        analysis = clean_text(analysis)
        answer = clean_text(answer)
        
        # Final safety: If answer still contains analysis content, try smarter split
        if answer and not analysis and len(answer) > 300:
            # Look for where analysis-like content (questions, reasoning) transitions to answer
            # Find last occurrence of questioning/reasoning phrases
            reasoning_indicators = [
                r'(.*?)(?:We can|We need|But|However|Given|Looking at|The user)(.*)',
            ]
            for pattern in reasoning_indicators:
                match = re.search(pattern, answer, re.DOTALL | re.IGNORECASE)
                if match:
                    potential_analysis = match.group(1).strip()
                    potential_answer = match.group(2).strip()
                    if len(potential_answer) > 50:  # Ensure answer has substantial content
                        analysis = potential_analysis
                        answer = potential_answer
                        break
        
        print(f"DEBUG: Extracted analysis length: {len(analysis)}, answer length: {len(answer)}")
        if analysis:
            print(f"DEBUG: Analysis preview: {analysis[:200]}...")
        if answer:
            print(f"DEBUG: Answer preview: {answer[:200]}...")
        
        return {
            "analysis": analysis,
            "answer": answer
        }
