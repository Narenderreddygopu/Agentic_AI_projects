import requests
import json

url = "https://openrouter.ai/api/v1/chat/completions"

API_KEY = "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"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Define tools that the AI can call
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Performs basic mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform: add, subtract, multiply, divide"
                    },
                    "a": {
                        "type": "number",
                        "description": "The first number"
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number"
                    }
                },
                "required": ["operation", "a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Gets the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city or location name"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Tool execution functions
def calculate(operation, a, b):
    """Execute calculation operations"""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b if b != 0 else "Error: Division by zero"
    else:
        return "Error: Unknown operation"

def get_weather(location):
    """Simulated weather function (would call real API)"""
    return f"Weather in {location}: 72°F, Sunny"

def execute_tool(tool_name, tool_input):
    """Execute the appropriate tool based on name"""
    if tool_name == "calculate":
        return calculate(**tool_input)
    elif tool_name == "get_weather":
        return get_weather(**tool_input)
    else:
        return "Unknown tool"

# Initial user message
messages = [
    {"role": "system", "content": "You are a helpful assistant with access to tools. Use them when appropriate."},
    {"role": "user", "content": "What can AI agents do? Also, can you calculate 25 + 37?"}
]

# Make request with tools
data = {
    "model": "openai/gpt-oss-120b:free",
    "messages": messages,
    "tools": tools
}

print("Sending request with tool definitions...")
response = requests.post(url, headers=headers, json=data)
result = response.json()

print("\n=== Initial Response ===")
print(json.dumps(result, indent=2))

# Handle tool calls if present
if "choices" in result and len(result["choices"]) > 0:
    choice = result["choices"][0]
    
    if choice.get("finish_reason") == "tool_calls" and "message" in choice:
        message = choice["message"]
        
        # Add assistant's response to messages
        messages.append({
            "role": "assistant",
            "content": message.get("content", ""),
            "tool_calls": message.get("tool_calls", [])
        })
        
        print("\n=== Tool Calls Detected ===")
        # Execute tool calls
        for tool_call in message.get("tool_calls", []):
            tool_name = tool_call["function"]["name"]
            tool_input = json.loads(tool_call["function"]["arguments"])
            
            print(f"Calling tool: {tool_name}")
            print(f"Input: {tool_input}")
            
            result = execute_tool(tool_name, tool_input)
            print(f"Result: {result}\n")
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_name,
                "content": str(result)
            })
        
        # Get follow-up response with tool results
        print("\n=== Requesting Follow-up Response ===")
        data["messages"] = messages
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        
        print(json.dumps(result, indent=2))
        
        # Extract and print final text response
        if "choices" in result and len(result["choices"]) > 0:
            final_message = result["choices"][0]["message"].get("content", "")
            print("\n=== Final Response ===")
            print(final_message)