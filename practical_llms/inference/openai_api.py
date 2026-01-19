#!/usr/bin/env python3
"""
OpenAI API Inference Examples
=============================

This tutorial demonstrates how to use the OpenAI API for various tasks:
1. Basic chat completion
2. Streaming responses
3. System prompts and roles
4. Function calling / Tool use
5. JSON mode
6. Vision (GPT-4V)

Prerequisites:
    pip install openai
    export OPENAI_API_KEY="your-key-here"
"""

import os
import json
from typing import Generator

# Check for API key early
if not os.environ.get("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not set. Set it with:")
    print('  export OPENAI_API_KEY="your-key-here"')


def get_client():
    """Get OpenAI client with error handling."""
    try:
        from openai import OpenAI
        return OpenAI()
    except ImportError:
        print("Please install openai: pip install openai")
        return None


# =============================================================================
# Example 1: Basic Chat Completion
# =============================================================================

def basic_chat_completion():
    """
    The simplest way to get a response from GPT.

    The messages list contains the conversation history.
    Each message has a 'role' (system/user/assistant) and 'content'.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Chat Completion")
    print("=" * 60)

    client = get_client()
    if not client:
        return

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-4", "gpt-4-turbo"
        messages=[
            {"role": "user", "content": "What is machine learning in one sentence?"}
        ],
        max_tokens=100,
        temperature=0.7,  # 0 = deterministic, 1 = creative
    )

    # Extract the response
    answer = response.choices[0].message.content
    print(f"\nResponse: {answer}")

    # Show usage statistics
    print(f"\nTokens used: {response.usage.total_tokens}")
    print(f"  - Prompt: {response.usage.prompt_tokens}")
    print(f"  - Completion: {response.usage.completion_tokens}")


# =============================================================================
# Example 2: Streaming Responses
# =============================================================================

def streaming_chat():
    """
    Stream responses token by token.

    Useful for:
    - Better UX (show text as it's generated)
    - Long responses (don't wait for full completion)
    - Chat applications
    """
    print("\n" + "=" * 60)
    print("Example 2: Streaming Response")
    print("=" * 60)

    client = get_client()
    if not client:
        return

    print("\nStreaming: ", end="", flush=True)

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Count from 1 to 10 slowly, with a brief pause description between each number."}
        ],
        stream=True,  # Enable streaming
        max_tokens=200,
    )

    # Process the stream
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content

    print("\n")  # Final newline


# =============================================================================
# Example 3: System Prompts and Multi-turn Conversation
# =============================================================================

def conversation_with_system_prompt():
    """
    Use system prompts to control the model's behavior.

    The system message sets the AI's personality, constraints, and context.
    This is how you create custom assistants.
    """
    print("\n" + "=" * 60)
    print("Example 3: System Prompts & Multi-turn Conversation")
    print("=" * 60)

    client = get_client()
    if not client:
        return

    # Define the system prompt
    system_prompt = """You are a helpful Python programming tutor.
    - Always explain concepts simply
    - Provide code examples when helpful
    - Be encouraging and patient
    - If asked about other topics, politely redirect to Python"""

    # Conversation history
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What's a list comprehension?"},
    ]

    # First turn
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300,
    )

    assistant_reply = response.choices[0].message.content
    print(f"\nUser: What's a list comprehension?")
    print(f"\nAssistant: {assistant_reply}")

    # Add assistant's response to history
    messages.append({"role": "assistant", "content": assistant_reply})

    # Second turn (follow-up question)
    messages.append({"role": "user", "content": "Can you show me a more complex example?"})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300,
    )

    print(f"\nUser: Can you show me a more complex example?")
    print(f"\nAssistant: {response.choices[0].message.content}")


# =============================================================================
# Example 4: Function Calling / Tool Use
# =============================================================================

def function_calling():
    """
    Let the model call functions/tools to get real-time data.

    The model doesn't execute functions - it returns structured data
    telling you which function to call with what arguments.
    You execute it and feed the result back.
    """
    print("\n" + "=" * 60)
    print("Example 4: Function Calling / Tool Use")
    print("=" * 60)

    client = get_client()
    if not client:
        return

    # Define available tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    # Simulated function (in reality, this would call a weather API)
    def get_weather(location: str, unit: str = "fahrenheit") -> dict:
        """Simulate a weather API call."""
        return {
            "location": location,
            "temperature": 72 if unit == "fahrenheit" else 22,
            "unit": unit,
            "condition": "sunny"
        }

    messages = [
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ]

    # First call - model decides to use a tool
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # Let model decide
    )

    assistant_message = response.choices[0].message
    print(f"\nUser: What's the weather like in San Francisco?")

    # Check if the model wants to call a function
    if assistant_message.tool_calls:
        tool_call = assistant_message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        print(f"\nModel wants to call: {function_name}({function_args})")

        # Execute the function
        if function_name == "get_weather":
            result = get_weather(**function_args)
            print(f"Function result: {result}")

        # Add the assistant's message and function result to conversation
        messages.append(assistant_message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

        # Get final response
        final_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )

        print(f"\nAssistant: {final_response.choices[0].message.content}")
    else:
        print(f"\nAssistant: {assistant_message.content}")


# =============================================================================
# Example 5: JSON Mode
# =============================================================================

def json_mode():
    """
    Force the model to output valid JSON.

    Useful for:
    - Structured data extraction
    - API responses
    - Configuration generation
    """
    print("\n" + "=" * 60)
    print("Example 5: JSON Mode")
    print("=" * 60)

    client = get_client()
    if not client:
        return

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You extract information and return it as JSON."
            },
            {
                "role": "user",
                "content": """Extract the following information from this text:

                "John Smith is a 32-year-old software engineer living in Seattle.
                He has 8 years of experience and specializes in Python and machine learning."

                Return JSON with fields: name, age, occupation, location, years_experience, skills (array)"""
            }
        ],
        response_format={"type": "json_object"},  # Force JSON output
    )

    result = response.choices[0].message.content
    print(f"\nExtracted JSON:")

    # Parse and pretty print
    parsed = json.loads(result)
    print(json.dumps(parsed, indent=2))


# =============================================================================
# Example 6: Embeddings
# =============================================================================

def embeddings_example():
    """
    Generate embeddings for semantic search and similarity.

    Embeddings are vector representations of text that capture meaning.
    Similar texts have similar embeddings (high cosine similarity).
    """
    print("\n" + "=" * 60)
    print("Example 6: Embeddings")
    print("=" * 60)

    client = get_client()
    if not client:
        return

    texts = [
        "I love programming in Python",
        "Python is my favorite programming language",
        "The weather is nice today",
    ]

    # Get embeddings
    response = client.embeddings.create(
        model="text-embedding-3-small",  # or text-embedding-3-large
        input=texts
    )

    embeddings = [item.embedding for item in response.data]

    print(f"\nGenerated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")

    # Compute cosine similarity
    import numpy as np

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print("\nCosine similarities:")
    print(f"  Text 1 vs Text 2 (similar meaning): {cosine_similarity(embeddings[0], embeddings[1]):.4f}")
    print(f"  Text 1 vs Text 3 (different topic):  {cosine_similarity(embeddings[0], embeddings[2]):.4f}")


# =============================================================================
# Example 7: Error Handling and Best Practices
# =============================================================================

def robust_api_call():
    """
    Production-ready API call with error handling and retries.
    """
    print("\n" + "=" * 60)
    print("Example 7: Robust API Calls")
    print("=" * 60)

    client = get_client()
    if not client:
        return

    import time
    from openai import APIError, RateLimitError, APIConnectionError

    def call_with_retry(messages, max_retries=3, base_delay=1):
        """Call API with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=100,
                    timeout=30,  # 30 second timeout
                )
                return response.choices[0].message.content

            except RateLimitError as e:
                # Rate limited - wait and retry
                delay = base_delay * (2 ** attempt)
                print(f"Rate limited. Waiting {delay}s before retry...")
                time.sleep(delay)

            except APIConnectionError as e:
                # Connection error - retry
                print(f"Connection error: {e}. Retrying...")
                time.sleep(base_delay)

            except APIError as e:
                # Other API error
                print(f"API error: {e}")
                raise

        raise Exception("Max retries exceeded")

    # Test the robust call
    result = call_with_retry([
        {"role": "user", "content": "Say 'Hello, World!'"}
    ])

    print(f"\nRobust call result: {result}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all examples."""
    print("=" * 60)
    print("OpenAI API Tutorial")
    print("=" * 60)

    if not os.environ.get("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        print("\nRunning in demo mode (showing code structure only)...")
        return

    try:
        # Run examples
        basic_chat_completion()
        streaming_chat()
        conversation_with_system_prompt()
        function_calling()
        json_mode()
        embeddings_example()
        robust_api_call()

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure your API key is valid and you have sufficient credits.")

    print("\n" + "=" * 60)
    print("Tutorial Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
