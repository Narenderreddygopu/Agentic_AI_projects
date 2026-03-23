# importing libraries
import requests 
import json

url = "https://openrouter.ai/api/v1/chat/completions"

API_KEY = "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"

headers = {
    "Authorization" : f"Bearer {API_KEY}",
    "Content-Type" : "application/json"
}


# json : {object : value}

tools = [
    {
        "type" : "function",
        "function" : {
            "name" : "calculate",
            "description" : "Performs basic mathematical calculations",
            "parameters" : {
                "type" : "object",
                "properties" : {
                    "operation" : {
                        "type" : "string",
                        "description" : "The operation to perform: add, subtract, multiply, divide"
                    },
                    "a" : {
                        "type" : "number",
                        "description" : "The first number"
                    },
                    "b" : {
                        "type" : "number",
                        "description" : "The second number"
                    }
                },
                "required" : ["operation", "a", "b"]
            }
        }
    },
    {
        "type" : "function",
        "function" : {
            "name" : "get_weather",
            "description" : "Gets the current weather for a location",
            "parameters" : {
                "type" : "object",
                "properties" : {
                    "location" : {
                        "type" : "string",
                        "description" : "The city or location name"
                    }
                },
                "required" : ["location"]
            }
        }
    }
]

def calculate(operation, a, b):
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b
    else:
        return "Invalid operation"
    
def get_weather(location):
    # This is a placeholder function. In a real implementation, you would call a weather API here.
    return f"The current weather in {location} is sunny with a temperature of 25°C."

def execute_tool(tool_name, tool_input):
    if tool_name == "calculate" :
        return calculate(**tool_input)
    elif tool_name == "get_weather" :
        return get_weather(**tool_input)
    else :
        return "Unknown tool"
messages = [
    {"role": "system", "content": "You are a helpful assistant. And the level of warmth while speaking keep it around mid level"},
    {"role": "user", "content": "What can AI agents do?"}
]

data = {
    "model" : "openai/gpt-oss-120b:free",
    "messages": messages,
    "tools" : tools
}


if __name__ == "__main__":
    response = requests.post(url, headers=headers, json=data)
    result = response.json()
    
    if "choices" in result and len(result["choices"]) > 0:
        choice = result["choices"][0]

        messages.append({
            "role" : "assistant",
            "content" : choice["message"]["content"]

        })
        print("Assistant response:")
        print(choice["message"]["content"])

        print(json.dumps(messages, indent=2))





