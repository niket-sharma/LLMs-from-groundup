import torch
import torch.nn.functional as F
from typing import List, Optional
import os


class GPTInference:
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> str:
        # Encode the prompt
        tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)

        # Generate
        with torch.no_grad():
            generated_tokens = self._generate_tokens(
                tokens, max_new_tokens, temperature, top_k, top_p, do_sample
            )

        # Decode the generated tokens
        generated_text = self.tokenizer.decode(generated_tokens[0].tolist())
        return generated_text

    def _generate_tokens(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        do_sample: bool,
    ) -> torch.Tensor:

        for _ in range(max_new_tokens):
            # Crop tokens if sequence is too long
            if tokens.size(1) > self.model.max_seq_len:
                tokens = tokens[:, -self.model.max_seq_len:]

            # Forward pass
            logits = self.model(tokens)
            logits = logits[:, -1, :]  # Get last token logits

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')

            # Sample or take argmax
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> List[str]:

        results = []
        for prompt in prompts:
            generated = self.generate(
                prompt, max_new_tokens, temperature, top_k, top_p, do_sample
            )
            results.append(generated)
        return results


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: Optional[str] = None,
) -> str:

    inference = GPTInference(model, tokenizer, device)
    return inference.generate(
        prompt, max_new_tokens, temperature, top_k, top_p
    )


def load_model_for_inference(model_class, checkpoint_path, tokenizer, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model (you might need to adjust this based on your saving format)
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create inference object
    inference = GPTInference(model, tokenizer, device)
    return inference