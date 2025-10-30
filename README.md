# OpenAI GPT-OSS-Safeguard-20B - Replicate Cog Implementation

A [Replicate](https://replicate.com) implementation of OpenAI's GPT-OSS-Safeguard-20B safety reasoning model using [Cog](https://github.com/replicate/cog). This implementation provides a bring-your-own-policy Trust & Safety AI that classifies text content based on customizable safety policies with full reasoning transparency.

## Features

- **Policy-Based Classification**: Provide your own written safety policy - no retraining needed
- **Reasoning Transparency**: Get complete access to the model's reasoning process in addition to classification decisions
- **Harmony Format**: Built-in support for OpenAI's harmony response format with separate analysis and answer channels
- **Configurable Reasoning**: Control reasoning depth with `reasoning_effort` parameter (low/medium/high)
- **Production Ready**: Clean output parsing with automatic extraction of analysis and final answers
- **Long Context**: Supports up to 131K token context length
- **Apache 2.0 License**: Free to use for commercial and non-commercial purposes

## Quick Start

### Prerequisites

- [Cog](https://github.com/replicate/cog) installed
- NVIDIA GPU with CUDA support (16GB+ VRAM recommended)
- Docker

### Test Locally

**Basic Classification:**
```bash
cog predict \
  -i prompt="Your bank details are needed to complete this transaction." \
  -i policy="Spam Policy: Classify content as VALID (no spam) or INVALID (spam). Spam includes unsolicited promotional content, deceptive messages, or phishing attempts."
```

**With Custom Reasoning Effort:**
```bash
cog predict \
  -i prompt="Check out my amazing product deal!" \
  -i policy="<your spam policy>" \
  -i reasoning_effort="high"
```

**Full Example with Policy:**
```bash
cog predict \
  -i prompt="NOW IS THE TIME TO CUT THE CORD AND JOIN. Where else will you get THE BEST that TV can offer for HALF the price?" \
  -i policy="Spam Policy (#SP)
GOAL: Identify spam. Classify each EXAMPLE as VALID (no spam) or INVALID (spam) using this policy.

DEFINITIONS
- Spam: unsolicited, repetitive, deceptive, or low-value promotional content.
- Bulk Messaging: Same or similar messages sent repeatedly.
- Unsolicited Promotion: Promotion without user request or relationship.

Allowed Content (SP0 – Non-Spam)
Content that is useful, contextual, or non-promotional.

Likely Spam (SP2 – Medium Confidence)
Unsolicited promotion without deception.

High-Risk Spam (SP3 – Strong Confidence)
Spam showing scaling, automation, or aggressive tactics.

Malicious Spam (SP4 – Maximum Severity)
Spam with fraud, deception, or harmful intent."
```

## Project Structure

```
.
├── predict.py              # Main Cog predictor implementation
├── cog.yaml               # Cog configuration
├── requirements.txt       # Python dependencies
├── checkpoints/           # Model files (downloaded automatically on first run)
│   ├── config.json        # Model configuration
│   ├── chat_template.jinja # Harmony format template
│   ├── tokenizer.json     # Tokenizer files
│   └── model-*.safetensors # Model weights
└── README.md              # This file
```

## Understanding the Harmony Format

GPT-OSS-Safeguard uses the [harmony response format](https://github.com/openai/harmony) to provide structured, auditable outputs. The model separates its response into two channels:

1. **Analysis Channel**: Contains the model's reasoning process, policy evaluation, and decision logic
2. **Final Channel**: Contains the formatted classification decision you specified

This dual-channel output enables Trust & Safety teams to:
- Understand why decisions were made
- Debug policy edge cases
- Audit classification quality
- Build trust in automated decisions

### Output Format

The `predict` method returns a dictionary with two fields:

```python
{
    "analysis": "The model's reasoning process and policy evaluation...",
    "answer": "The final classification decision or formatted output..."
}
```

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | String | "Your bank details are needed..." | Text content to evaluate against the policy |
| `policy` | String | "" | Safety policy or rules to evaluate the prompt against. Should include definitions, examples, and output format instructions |
| `max_new_tokens` | Integer | 512 | Maximum number of tokens to generate (1-8192) |
| `temperature` | Float | 1.0 | Sampling temperature (0.0-2.0, 0 = greedy decoding) |
| `top_p` | Float | 1.0 | Nucleus sampling top-p (0.0-1.0) |
| `repetition_penalty` | Float | 1.0 | Penalty for repetitive text (0.8-2.0, 1.0 = no penalty) |
| `reasoning_effort` | String | "medium" | Reasoning depth: "low" (faster), "medium" (balanced), or "high" (deeper analysis) |

### Reasoning Effort Levels

- **`low`**: Faster responses for straightforward classifications. Use when latency is critical and content is clearly compliant or non-compliant.
- **`medium`**: Balanced reasoning (default). Good for most production use cases.
- **`high`**: Deeper analysis for complex edge cases. Use when you need thorough policy tracing, handling nuanced content, or debugging borderline decisions.

## Writing Effective Policy Prompts

The model's performance depends heavily on well-crafted policies. Key guidelines:

1. **Define Terms Clearly**: Explicitly define key terms and concepts
2. **Provide Examples**: Include examples of compliant and non-compliant content
3. **Specify Output Format**: Tell the model exactly how to format its decision
4. **Use Hierarchical Structure**: Organize policies with clear sections and subsections
5. **Include Edge Cases**: Anticipate and provide guidance for borderline cases

### Policy Template

```markdown
## Policy Definitions

### Key Terms

**[Term 1]**: [Definition]

**[Term 2]**: [Definition]

## Content Classification Rules

### VIOLATES Policy (Label: 1)

Content that:
- [Violation 1]
- [Violation 2]

### DOES NOT Violate Policy (Label: 0)

Content that is:
- [Acceptable 1]
- [Acceptable 2]

## Examples

### Example 1 (Label: 1)
**Content**: "[Example]"
**Expected Response**: [What you expect]

### Example 2 (Label: 0)
**Content**: "[Example]"
**Expected Response**: [What you expect]
```

For detailed guidance on writing effective policies, see the [OpenAI Cookbook Guide](https://cookbook.openai.com/articles/gpt-oss-safeguard-guide).

## Example Use Cases

### Content Moderation

```bash
cog predict \
  -i prompt="Check out these amazing deals! Limited time offer!" \
  -i policy="Spam detection policy" \
  -i reasoning_effort="medium"
```

### Trust & Safety Classification

```bash
cog predict \
  -i prompt="User-submitted content to evaluate" \
  -i policy="Your platform's community guidelines" \
  -i reasoning_effort="high"
```

### Policy Testing

```bash
cog predict \
  -i prompt="Test case content" \
  -i policy="New policy draft for testing" \
  -i reasoning_effort="high"
```

## Development

### Building the Model

```bash
# Build the Cog container
cog build

# Test the build locally
cog predict \
  -i prompt="Test prompt" \
  -i policy="Test policy"
```

### Pushing to Replicate

```bash
# Login to Replicate
cog login

# Push to your Replicate account
cog push r8.im/your-username/gpt-oss-safeguard-20b
```

### Model Weights

Model weights are automatically downloaded on first run using `pget`:
- **Model**: Downloaded to `./checkpoints/` from Replicate's CDN
- **Size**: ~40GB (quantized MXFP4 format)

For local development with pre-downloaded weights, place them in the `checkpoints/` directory.

## Model Information

**Base Model**: [OpenAI GPT-OSS-Safeguard-20B](https://huggingface.co/openai/gpt-oss-safeguard-20b)
- **Parameters**: 21B total, 3.6B active (Mixture of Experts)
- **Context Length**: 131K tokens
- **Quantization**: MXFP4
- **Base Architecture**: GPT-OSS-20B fine-tuned for safety reasoning
- **License**: Apache 2.0

**Architecture Details**:
- Mixture of Experts (MoE) with 32 experts, 4 active per token
- Sliding window attention (128 tokens) alternating with full attention
- YaRN rope scaling for extended context
- Trained on harmony response format

## Technical Details

### System Requirements

- **GPU**: NVIDIA GPU with CUDA 12.4 support
- **VRAM**: 16GB+ recommended (model uses MXFP4 quantization)
- **Python**: 3.11
- **System Packages**: None required (all dependencies installed via pip)

### Dependencies

Key dependencies (see `requirements.txt` for full list):
- `torch>=2.1.0` - PyTorch framework
- `transformers>=4.45.0` - HuggingFace Transformers
- `accelerate>=0.25.0` - Model acceleration
- `tokenizers>=0.15.0` - Fast tokenization

### Harmony Format Implementation

The implementation properly handles:
- Developer role messages (for policy instructions)
- User role messages (for content to evaluate)
- Reasoning effort control via template kwargs
- Dual-channel output parsing (analysis + final answer)
- Special token handling (`<|start|>`, `<|end|>`, `<|return|>`, etc.)

## Integration Examples

### Trust & Safety Pipeline

Use as a reasoning layer after fast pre-filters:

```python
# Pseudocode example
if fast_classifier.indicates_risk(content):
    result = gpt_oss_safeguard.predict(
        prompt=content,
        policy=platform_policy,
        reasoning_effort="high"
    )
    if result["answer"] == "VIOLATES":
        escalate_for_review(result["analysis"])
```

### Policy A/B Testing

Test alternative policy definitions without retraining:

```python
# Test Policy A
result_a = model.predict(prompt=content, policy=policy_a)

# Test Policy B  
result_b = model.predict(prompt=content, policy=policy_b)

# Compare results
compare_decisions(result_a, result_b)
```

## Resources

- **Try Online**: [HuggingFace Space](https://huggingface.co/spaces/openai/gpt-oss-safeguard-20b)
- **User Guide**: [OpenAI Cookbook Guide](https://cookbook.openai.com/articles/gpt-oss-safeguard-guide)
- **Model Card**: [arXiv Paper](https://arxiv.org/abs/2508.10925)
- **OpenAI Blog**: [Introducing GPT-OSS-Safeguard](https://openai.com/index/introducing-gpt-oss-safeguard/)
- **Harmony Format**: [GitHub Repository](https://github.com/openai/harmony)
- **ROOST Community**: [Robust Open Online Safety Tools](http://roost.tools/)

## License

This implementation is provided for use with the OpenAI GPT-OSS-Safeguard-20B model, which is released under the **Apache 2.0 License**. See the [model repository](https://huggingface.co/openai/gpt-oss-safeguard-20b) for full license details.

## Contributing

Contributions are welcome! Please ensure:
- Code follows the existing style
- All features are tested with `cog predict`
- Documentation is updated for new features
- Policy examples follow best practices from the cookbook guide

## Support

For issues with this Replicate/Cog implementation, please open a GitHub issue.

For questions about:
- **Model usage**: See the [OpenAI Cookbook Guide](https://cookbook.openai.com/articles/gpt-oss-safeguard-guide)
- **Policy writing**: Refer to the guide's "Writing Effective Policy Prompts" section
- **Harmony format**: Check the [harmony repository](https://github.com/openai/harmony)

## Acknowledgments

- Model developed by OpenAI
- Cog framework by Replicate
- Implementation follows OpenAI's harmony format specification
- Part of the ROOST (Robust Open Online Safety Tools) Model Community

