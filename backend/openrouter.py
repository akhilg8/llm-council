"""OpenRouter API client for making LLM requests."""

import httpx
import asyncio
import random
from typing import List, Dict, Any, Optional
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL

# Set to True to enable mocking
MOCK_MODE = True

async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    if MOCK_MODE:
        return await mock_response(model, messages)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            message = data['choices'][0]['message']

            return {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details')
            }

    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    # Create tasks for all models
    tasks = [query_model(model, messages) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}


async def mock_response(model: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Generate a mock response based on the input messages."""

    # Simulate network delay
    await asyncio.sleep(random.uniform(0.5, 2.0))

    last_message = messages[-1]['content']

    # Check if this is a title generation request
    if "Generate a very short title" in last_message:
        # Extract the question part if possible, or just return a generic title
        return {
            'content': "Cyberpunk Chat Session",
            'reasoning_details': None
        }

    # Check if this is a Chairman synthesis request (Stage 3)
    # Check this BEFORE ranking because Stage 3 prompt includes "FINAL RANKING" text from previous stages
    if "You are the Chairman of an LLM Council" in last_message:
        return {
            'content': f"As the Chairman, I have reviewed the input from the council. \n\nBased on the collective wisdom, the answer to your query is: \n\nThis is a mocked synthesis response for the query. The council has debated and concluded that this is the best course of action. \n\nMocked Insights:\n- Point 1: The council agrees on X.\n- Point 2: There was some debate on Y.\n- Point 3: Ultimately, Z is the recommended path.",
            'reasoning_details': None
        }

    # Check if this is a ranking request (Stage 2)
    if "FINAL RANKING:" in last_message:
        # We need to find which response labels are available
        # The prompt usually contains "Response A", "Response B", etc.
        import re
        labels = re.findall(r'Response ([A-Z]):', last_message)
        if not labels:
            # Fallback if regex fails (shouldn't happen with our prompt structure)
            labels = ['A', 'B', 'C', 'D', 'E']

        # Shuffle labels to create random rankings
        shuffled_labels = labels.copy()
        random.shuffle(shuffled_labels)

        ranking_text = "Here is my evaluation of the responses:\n\n"
        for label in labels:
            ranking_text += f"Response {label} makes some good points but could be improved.\n"

        ranking_text += "\nFINAL RANKING:\n"
        for i, label in enumerate(shuffled_labels, 1):
            ranking_text += f"{i}. Response {label}\n"

        return {
            'content': ranking_text,
            'reasoning_details': None
        }

    # Default: Stage 1 response (answering the user query)
    # We can try to give a slightly different response based on the model name to make it look realistic
    model_short = model.split('/')[-1]
    return {
        'content': f"[{model_short}] This is a mocked response to: '{last_message[:50]}...'. \n\nI believe the answer involves analyzing the cyber-structures of the data streams. The neon lights of the city illuminate the truth.",
        'reasoning_details': None
    }
