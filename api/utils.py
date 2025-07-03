import requests
from typing import Literal
import time
import json

def check_ollama_availability():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def generate_tag_for_passion(passion: str, model: Literal['llama3.2:latest', 'all-minilm']):
    if not check_ollama_availability():
        raise ConnectionError("Ollama is not running. Please start Ollama first using 'ollama serve'")

    prompt = f"""
        You are a travel expert.
        You are given a passion and you need to generate a tag for it.
        The tag should be a single word or phrase that captures the essence of the passion.
        The tag should be 1-3 words.
        The tag should be no more than 10 characters.
        ---
        passion: {passion}
        ---
        example: 

        input: "I love eating food so much, it gives me so much joy"
        response: "food"
        ---
        response: 
    """
    
    if model == 'llama3.2:latest':
        json_data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    else:
        json_data = {
            "model": model,
            "prompt": prompt,
        }

    try:
        res = requests.post(
                "http://localhost:11434/api/generate", 
                json=json_data,
                stream=False
            )
        res.raise_for_status()
        if model == 'mistral:7b':
            # Handle newline-delimited JSON response
            full_response = ""
            for line in res.text.strip().split('\n'):
                if line:
                    try:
                        response_data = json.loads(line)
                        if 'response' in response_data:
                            full_response += response_data['response']
                    except json.JSONDecodeError:
                        continue
        
            return {"response": full_response.strip()}
        else:
            data = res.json()
            return data
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to connect to Ollama: {str(e)}")


if __name__ == "__main__":
    model = ['llama3.2:latest', 'mistral:7b']

    try:
        data = generate_tag_for_passion("I love eating food so much, it gives me so much joy", model[0])
        print("Generated tag:", data['response'])
    except ConnectionError as e:
        print(f"Error: {e}")
        print("Please make sure Ollama is running by executing 'ollama serve' in a terminal")