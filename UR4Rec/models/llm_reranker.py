"""
LLM-based Reranker.

Integrates with Large Language Models to perform reranking based on
retrieved user preferences.
"""
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union, Tuple
import json
from abc import ABC, abstractmethod


class LLMInterface(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for batch of prompts."""
        pass


class OpenAILLM(LLMInterface):
    """OpenAI API interface."""

    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("Please install openai package: pip install openai")

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate response from OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> List[str]:
        """Generate responses for batch of prompts."""
        return [self.generate(p, max_tokens, temperature) for p in prompts]


class AnthropicLLM(LLMInterface):
    """Anthropic Claude API interface."""

    def __init__(self, model: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("Please install anthropic package: pip install anthropic")

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate response from Anthropic API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> List[str]:
        """Generate responses for batch of prompts."""
        return [self.generate(p, max_tokens, temperature) for p in prompts]


class LocalLLM(LLMInterface):
    """Local LLM interface using HuggingFace transformers."""

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", device: str = "cuda"):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
            self.device = device
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate response using local model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        response = response[len(prompt):].strip()
        return response

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> List[str]:
        """Generate responses for batch of prompts."""
        return [self.generate(p, max_tokens, temperature) for p in prompts]


class LLMReranker(nn.Module):
    """
    LLM-based reranker that uses retrieved user preferences.
    """

    def __init__(
        self,
        llm_backend: str = "openai",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        item_metadata: Optional[Dict] = None,
        prompt_template: Optional[str] = None
    ):
        """
        Args:
            llm_backend: Backend to use ('openai', 'anthropic', 'local')
            model_name: Name of the model to use
            api_key: API key for commercial providers
            item_metadata: Dictionary mapping item IDs to metadata
            prompt_template: Custom prompt template
        """
        super().__init__()

        # Initialize LLM
        if llm_backend == "openai":
            self.llm = OpenAILLM(model=model_name or "gpt-3.5-turbo", api_key=api_key)
        elif llm_backend == "anthropic":
            self.llm = AnthropicLLM(model=model_name or "claude-3-sonnet-20240229", api_key=api_key)
        elif llm_backend == "local":
            self.llm = LocalLLM(model_name=model_name or "meta-llama/Llama-2-7b-chat-hf")
        else:
            raise ValueError(f"Unknown LLM backend: {llm_backend}")

        self.item_metadata = item_metadata or {}
        self.prompt_template = prompt_template or self._default_prompt_template()

    def _default_prompt_template(self) -> str:
        """Default prompt template for reranking."""
        return """You are a recommendation system expert. Given a user's interaction history and candidate items, please rerank the candidates based on user preferences.

User Interaction History:
{history}

Candidate Items to Rerank:
{candidates}

Retrieved User Preferences:
{preferences}

Please analyze the user's preferences and rerank the candidate items. Output ONLY a JSON list of item IDs in the reranked order, from most relevant to least relevant.

Example output format: [5, 2, 8, 1, 3, 7, 4, 6]

Reranked items:"""

    def _format_item(self, item_id: int) -> str:
        """Format item for prompt."""
        if item_id in self.item_metadata:
            metadata = self.item_metadata[item_id]
            return f"Item {item_id}: {metadata}"
        return f"Item {item_id}"

    def _format_history(self, history: List[int]) -> str:
        """Format user history for prompt."""
        formatted = [self._format_item(item_id) for item_id in history[-20:]]  # Last 20 items
        return "\n".join(f"{i+1}. {item}" for i, item in enumerate(formatted))

    def _format_candidates(self, candidates: List[int]) -> str:
        """Format candidate items for prompt."""
        formatted = [self._format_item(item_id) for item_id in candidates]
        return "\n".join(f"{i+1}. {item}" for i, item in enumerate(formatted))

    def _format_preferences(self, preference_scores: torch.Tensor, candidates: List[int]) -> str:
        """Format retrieved preferences for prompt."""
        scores = preference_scores.tolist()
        sorted_pairs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        formatted = []
        for item_id, score in sorted_pairs[:10]:  # Top 10 preferences
            formatted.append(f"- {self._format_item(item_id)} (preference score: {score:.3f})")

        return "\n".join(formatted)

    def _parse_llm_output(self, output: str, candidates: List[int]) -> List[int]:
        """Parse LLM output to extract ranked list."""
        try:
            # Try to extract JSON list
            import re
            json_match = re.search(r'\[[\d,\s]+\]', output)
            if json_match:
                ranked_items = json.loads(json_match.group())
                # Validate that all items are in candidates
                valid_items = [item for item in ranked_items if item in candidates]
                # Add any missing candidates at the end
                missing = [item for item in candidates if item not in valid_items]
                return valid_items + missing
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: return original order
        return candidates

    def rerank(
        self,
        user_history: Union[List[int], torch.Tensor],
        candidates: Union[List[int], torch.Tensor],
        preference_scores: torch.Tensor,
        temperature: float = 0.3
    ) -> List[int]:
        """
        Rerank candidates using LLM.

        Args:
            user_history: User's interaction history
            candidates: Candidate items to rerank
            preference_scores: Scores from preference retriever
            temperature: Sampling temperature for LLM

        Returns:
            Reranked list of item IDs
        """
        # Convert to lists if tensors
        if isinstance(user_history, torch.Tensor):
            user_history = user_history.tolist()
        if isinstance(candidates, torch.Tensor):
            candidates = candidates.tolist()

        # Remove padding (zeros)
        user_history = [x for x in user_history if x != 0]
        candidates = [x for x in candidates if x != 0]

        # Format prompt
        prompt = self.prompt_template.format(
            history=self._format_history(user_history),
            candidates=self._format_candidates(candidates),
            preferences=self._format_preferences(preference_scores, candidates)
        )

        # Generate ranking
        output = self.llm.generate(prompt, max_tokens=200, temperature=temperature)

        # Parse output
        ranked_items = self._parse_llm_output(output, candidates)

        return ranked_items

    def batch_rerank(
        self,
        user_histories: List[List[int]],
        candidates_list: List[List[int]],
        preference_scores_list: List[torch.Tensor],
        temperature: float = 0.3
    ) -> List[List[int]]:
        """
        Rerank multiple batches.

        Args:
            user_histories: List of user histories
            candidates_list: List of candidate lists
            preference_scores_list: List of preference score tensors
            temperature: Sampling temperature

        Returns:
            List of reranked item lists
        """
        results = []
        for history, candidates, scores in zip(
            user_histories, candidates_list, preference_scores_list
        ):
            ranked = self.rerank(history, candidates, scores, temperature)
            results.append(ranked)

        return results


if __name__ == "__main__":
    print("Testing LLM Reranker...")

    # Sample data
    user_history = [101, 205, 303, 412, 567, 689, 701]
    candidates = [801, 802, 803, 804, 805]
    preference_scores = torch.tensor([0.8, 0.6, 0.9, 0.4, 0.7])

    # Sample item metadata
    item_metadata = {
        101: "Book: Python Programming",
        205: "Book: Machine Learning Basics",
        303: "Book: Deep Learning",
        801: "Book: Natural Language Processing",
        802: "Book: Computer Vision",
        803: "Book: Reinforcement Learning",
        804: "Book: Data Structures",
        805: "Book: Advanced Deep Learning"
    }

    print("\nNote: To test with actual LLM, set up API keys:")
    print("- OpenAI: export OPENAI_API_KEY=your_key")
    print("- Anthropic: export ANTHROPIC_API_KEY=your_key")
    print("\nCurrently creating mock reranker...")

    # Create mock reranker (no actual LLM calls)
    print(f"\nInput:")
    print(f"  User history: {user_history}")
    print(f"  Candidates: {candidates}")
    print(f"  Preference scores: {preference_scores.tolist()}")

    print(f"\nExpected reranking would prioritize items with higher preference scores")
    print(f"Example output: {sorted(candidates, key=lambda x: preference_scores[candidates.index(x)], reverse=True)}")
