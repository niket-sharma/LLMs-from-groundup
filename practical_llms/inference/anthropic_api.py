#!/usr/bin/env python3
"""
Anthropic (Claude) API Inference Examples
=========================================

This tutorial demonstrates how to use the Anthropic API with Claude:
1. Basic message completion
2. Streaming responses
3. System prompts
4. Multi-turn conversations
5. Tool use (function calling)
6. Vision (Claude 3)

Prerequisites:
    pip install anthropic
    export ANTHROPIC_API_KEY="your-key-here"
"""

import os
import json
from typing import List, Dict, Any

# Check for API key early
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("Warning: ANTHROPIC_API_KEY not set. Set it with:")
    print('  export ANTHROPIC_API_KEY="your-key-here"')


def get_client():
    """Get Anthropic client with error handling."""
    try:
        from anthropic import Anthropic
        return Anthropic()
    except ImportError:
        print("Please install anthropic: pip install anthropic")
        return None


# =============================================================================
# Example 1: Basic Message Completion
# =============================================================================

def basic_completion():
    """
    The simplest way to get a response from Claude.

    Key differences from OpenAI:
    - Uses 'messages.create' instead of 'chat.completions.create'
    - System prompt is a separate parameter, not in messages
    - max_tokens is required
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Message Completion")
    print("=" * 60)

    client = get_client()
    if not client:
        return

    response = client.messages.create(
        model="claude-3-haiku-20240307",  # Fast and cheap
        # Other options:
        # - claude-3-sonnet-20240229 (balanced)
        # - claude-3-opus-20240229 (most capable)
        # - claude-3-5-sonnet-20241022 (latest)
        max_tokens=100,
        messages=[
            {"role": "user", "content": "What is machine learning in one sentence?"}
        ]
    )

    # Extract the response
    answer = response.content[0].text
    print(f"\nResponse: {answer}")

    # Show usage statistics
    print(f"\nTokens used:")
    print(f"  - Input: {response.usage.input_tokens}")
    print(f"  - Output: {response.usage.output_tokens}")
    print(f"  - Stop reason: {response.stop_reason}")


# =============================================================================
# Example 2: Streaming Responses
# =============================================================================

def streaming_response():
    """
    Stream responses token by token.

    Claude's streaming uses server-sent events (SSE).
    Each chunk contains a delta with the new text.
    """
    print("\n" + "=" * 60)
    print("Example 2: Streaming Response")
    print("=" * 60)

    client = get_client()
    if not client:
        return

    print("\nStreaming: ", end="", flush=True)

    with client.messages.stream(
        model="claude-3-haiku-20240307",
        max_tokens=200,
        messages=[
            {"role": "user", "content": "Count from 1 to 10, with a fun fact about each number."}
        ]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    print("\n")


# =============================================================================
# Example 3: System Prompts
# =============================================================================

def system_prompts():
    """
    Use system prompts to control Claude's behavior.

    In Anthropic's API, the system prompt is a separate parameter,
    not part of the messages array.
    """
    print("\n" + "=" * 60)
    print("Example 3: System Prompts")
    print("=" * 60)

    client = get_client()
    if not client:
        return

    system_prompt = """You are a pirate captain who loves to explain technology.
    - Always speak in pirate dialect
    - Use nautical metaphors for technical concepts
    - Be enthusiastic and encouraging
    - End responses with a pirate saying"""

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=300,
        system=system_prompt,  # System prompt goes here
        messages=[
            {"role": "user", "content": "Explain how neural networks work."}
        ]
    )

    print(f"\nUser: Explain how neural networks work.")
    print(f"\nClaude (as pirate): {response.content[0].text}")


# =============================================================================
# Example 4: Multi-turn Conversation
# =============================================================================

def multi_turn_conversation():
    """
    Maintain context across multiple turns.

    Like OpenAI, you need to include the full conversation history.
    """
    print("\n" + "=" * 60)
    print("Example 4: Multi-turn Conversation")
    print("=" * 60)

    client = get_client()
    if not client:
        return

    conversation = []

    # Helper function to chat
    def chat(user_message: str) -> str:
        conversation.append({"role": "user", "content": user_message})

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=300,
            system="You are a helpful coding assistant.",
            messages=conversation
        )

        assistant_message = response.content[0].text
        conversation.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    # Have a multi-turn conversation
    print("\n--- Turn 1 ---")
    print("User: What's a decorator in Python?")
    print(f"Claude: {chat('What is a decorator in Python?')}")

    print("\n--- Turn 2 ---")
    print("User: Show me an example")
    print(f"Claude: {chat('Show me an example')}")

    print("\n--- Turn 3 ---")
    print("User: How would I use this to measure function execution time?")
    print(f"Claude: {chat('How would I use this to measure function execution time?')}")


# =============================================================================
# Example 5: Tool Use (Function Calling)
# =============================================================================

def tool_use():
    """
    Let Claude call tools/functions.

    Claude can decide which tools to use and with what arguments.
    You execute the tools and return results.
    """
    print("\n" + "=" * 60)
    print("Example 5: Tool Use (Function Calling)")
    print("=" * 60)

    client = get_client()
    if not client:
        return

    # Define available tools
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
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
        },
        {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '2 + 2'"
                    }
                },
                "required": ["expression"]
            }
        }
    ]

    # Simulated functions
    def get_weather(location: str, unit: str = "fahrenheit") -> dict:
        return {"location": location, "temperature": 72, "unit": unit, "condition": "sunny"}

    def calculate(expression: str) -> dict:
        try:
            result = eval(expression)  # Note: eval is unsafe in production!
            return {"expression": expression, "result": result}
        except:
            return {"error": "Invalid expression"}

    # First request
    messages = [
        {"role": "user", "content": "What's the weather in Tokyo and what's 15% of 85?"}
    ]

    print("\nUser: What's the weather in Tokyo and what's 15% of 85?")

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    # Process tool calls
    if response.stop_reason == "tool_use":
        # Add Claude's response to messages
        messages.append({"role": "assistant", "content": response.content})

        # Process each tool use
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\nClaude wants to call: {block.name}({block.input})")

                # Execute the tool
                if block.name == "get_weather":
                    result = get_weather(**block.input)
                elif block.name == "calculate":
                    result = calculate(**block.input)
                else:
                    result = {"error": "Unknown tool"}

                print(f"Tool result: {result}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result)
                })

        # Send tool results back
        messages.append({"role": "user", "content": tool_results})

        # Get final response
        final_response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )

        print(f"\nClaude: {final_response.content[0].text}")
    else:
        print(f"\nClaude: {response.content[0].text}")


# =============================================================================
# Example 6: Structured Output with XML Tags
# =============================================================================

def structured_output():
    """
    Get structured output using XML tags.

    Claude responds well to XML-style prompting.
    This is an alternative to JSON mode.
    """
    print("\n" + "=" * 60)
    print("Example 6: Structured Output with XML Tags")
    print("=" * 60)

    client = get_client()
    if not client:
        return

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": """Analyze this product review and provide structured output:

Review: "I bought this laptop last month. The screen is absolutely gorgeous
and the battery lasts all day. However, the keyboard feels a bit mushy
and it gets warm during heavy use. Overall, I'm satisfied with my purchase."

Please respond with:
<analysis>
  <sentiment>positive/negative/mixed</sentiment>
  <pros>
    <item>pro 1</item>
    <item>pro 2</item>
  </pros>
  <cons>
    <item>con 1</item>
    <item>con 2</item>
  </cons>
  <overall_rating>1-5</overall_rating>
  <summary>brief summary</summary>
</analysis>"""
            }
        ]
    )

    print("\nStructured Analysis:")
    print(response.content[0].text)


# =============================================================================
# Example 7: Vision (Image Analysis)
# =============================================================================

def vision_example():
    """
    Analyze images with Claude 3.

    Claude can understand images passed as:
    - Base64-encoded data
    - URLs (public images)
    """
    print("\n" + "=" * 60)
    print("Example 7: Vision (Image Analysis)")
    print("=" * 60)

    client = get_client()
    if not client:
        return

    # Using a public image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url,
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in detail. What do you see?"
                    }
                ]
            }
        ]
    )

    print(f"\nImage URL: {image_url}")
    print(f"\nClaude's description: {response.content[0].text}")


# =============================================================================
# Example 8: Error Handling
# =============================================================================

def robust_api_call():
    """
    Production-ready API calls with error handling.
    """
    print("\n" + "=" * 60)
    print("Example 8: Robust API Calls")
    print("=" * 60)

    client = get_client()
    if not client:
        return

    import time
    from anthropic import APIError, RateLimitError, APIConnectionError

    def call_with_retry(messages, max_retries=3, base_delay=1):
        """Call API with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=100,
                    messages=messages,
                )
                return response.content[0].text

            except RateLimitError as e:
                delay = base_delay * (2 ** attempt)
                print(f"Rate limited. Waiting {delay}s before retry...")
                time.sleep(delay)

            except APIConnectionError as e:
                print(f"Connection error: {e}. Retrying...")
                time.sleep(base_delay)

            except APIError as e:
                print(f"API error: {e}")
                raise

        raise Exception("Max retries exceeded")

    result = call_with_retry([
        {"role": "user", "content": "Say 'Hello from Claude!'"}
    ])

    print(f"\nRobust call result: {result}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all examples."""
    print("=" * 60)
    print("Anthropic (Claude) API Tutorial")
    print("=" * 60)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nError: ANTHROPIC_API_KEY environment variable not set.")
        print("Please set it with: export ANTHROPIC_API_KEY='your-key-here'")
        print("\nRunning in demo mode (showing code structure only)...")
        return

    try:
        basic_completion()
        streaming_response()
        system_prompts()
        multi_turn_conversation()
        tool_use()
        structured_output()
        # vision_example()  # Uncomment if you want to test vision
        robust_api_call()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Tutorial Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
