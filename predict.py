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
        # The model outputs: <|start|>assistant<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...<|return|>
        analysis = ""
        answer = ""
        
        # Try to extract analysis (reasoning) and final answer
        analysis_match = re.search(r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>', output_text, re.DOTALL)
        if analysis_match:
            analysis = analysis_match.group(1).strip()
        
        # Extract final answer
        final_match = re.search(r'<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)(?:<\|return\||<\|end\||$)', output_text, re.DOTALL)
        if final_match:
            answer = final_match.group(1).strip()
        elif not analysis_match:
            # Fallback: if no format markers, use entire output as answer
            answer = output_text.strip()
            # Clean up any remaining special tokens
            answer = re.sub(r'<\|[^|]+\|>', '', answer).strip()
        
        # Clean up the answer by removing special tokens
        answer = re.sub(r'<\|[^|]+\|>', '', answer).strip()
        analysis = re.sub(r'<\|[^|]+\|>', '', analysis).strip()
        
        return {
            "analysis": analysis,
            "answer": answer
        }
