import requests
import json

def test_ollama():
    try:
        # Test GET request
        response = requests.get("http://127.0.0.1:11434/api/tags")
        print(f"GET Response status: {response.status_code}")
        print(f"GET Response content: {response.text[:200]}...")

        # Test POST request
        data = {
            "model": "deepseek-r1:32b",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        response = requests.post("http://127.0.0.1:11434/api/chat", json=data)
        print(f"POST Response status: {response.status_code}")
        print(f"POST Response content: {response.text[:200]}...")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_ollama()
    print(f"Test {'succeeded' if success else 'failed'}")