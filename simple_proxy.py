import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class SimpleOllamaProxy(BaseHTTPRequestHandler):
    def _set_headers(self, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers()
        
    def do_GET(self):
        if self.path == '/':
            # Health check
            try:
                response = requests.get('http://127.0.0.1:11434/api/tags')
                if response.status_code == 200:
                    self._set_headers()
                    self.wfile.write(json.dumps({"status": "ok", "ollama": "connected"}).encode())
                else:
                    self._set_headers(502)
                    self.wfile.write(json.dumps({"error": "Failed to connect to Ollama"}).encode())
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode())
                
        elif self.path == '/v1/models':
            try:
                response = requests.get('http://127.0.0.1:11434/api/tags')
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    openai_models = {
                        "object": "list",
                        "data": [
                            {
                                "id": model.get("name", "unknown"),
                                "object": "model",
                                "created": 1677610602,
                                "owned_by": "ollama"
                            }
                            for model in models
                        ]
                    }
                    self._set_headers()
                    self.wfile.write(json.dumps(openai_models).encode())
                else:
                    self._set_headers(response.status_code)
                    self.wfile.write(response.text.encode())
            except Exception as e:
                print(f"Error in /v1/models: {e}")
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

    def do_POST(self):
        if self.path == '/v1/chat/completions':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                print(f"Received request: {data}")
                
                # Convert to Ollama format
                ollama_data = {
                    "model": data.get("model", "deepseek-r1:32b").split('/')[-1],
                    "messages": data.get("messages", []),
                    "stream": False  # Important! Disable streaming
                }
                
                print(f"Sending to Ollama: {ollama_data}")
                
                # Send to Ollama
                ollama_response = requests.post('http://127.0.0.1:11434/api/chat', json=ollama_data)
                
                if ollama_response.status_code == 200:
                    try:
                        ollama_result = ollama_response.json()
                        print(f"Got Ollama result type: {type(ollama_result)}")
                        print(f"Got Ollama result keys: {ollama_result.keys() if isinstance(ollama_result, dict) else 'not a dict'}")
                        
                        # Extract content from Ollama response
                        content = ""
                        if isinstance(ollama_result, dict):
                            message = ollama_result.get("message", {})
                            if isinstance(message, dict):
                                content = message.get("content", "")
                        
                        # Convert to OpenAI format
                        openai_response = {
                            "id": "chatcmpl-123456",
                            "object": "chat.completion",
                            "created": 1679351277,
                            "model": ollama_data["model"],
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": content
                                    },
                                    "finish_reason": "stop",
                                    "index": 0
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0
                            }
                        }
                        
                        print(f"Sending OpenAI response: {json.dumps(openai_response)[:200]}...")
                        self._set_headers()
                        self.wfile.write(json.dumps(openai_response).encode())
                    except Exception as e:
                        print(f"Error processing Ollama response: {e}")
                        print(f"Response text: {ollama_response.text[:500]}")
                        self._set_headers(500)
                        self.wfile.write(json.dumps({"error": f"Error processing Ollama response: {str(e)}"}).encode())
                else:
                    print(f"Ollama error: {ollama_response.status_code} {ollama_response.text}")
                    self._set_headers(ollama_response.status_code)
                    self.wfile.write(ollama_response.text.encode())
            except Exception as e:
                print(f"Error in chat completions: {e}")
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, SimpleOllamaProxy)
    print(f'Starting proxy server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()