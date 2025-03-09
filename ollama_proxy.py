from fastapi import FastAPI, Request, Response
import httpx
import uvicorn
import json
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

app = FastAPI()
# Use 127.0.0.1 instead of localhost for more reliable connections
client = httpx.AsyncClient(timeout=60.0)
OLLAMA_URL = "http://127.0.0.1:11434"

@app.get("/")
async def root():
    try:
        # Test connection to Ollama
        logger.info(f"Testing connection to Ollama at {OLLAMA_URL}/api/tags")
        response = await client.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            return {"status": "Proxy server is running", "ollama_connection": "OK", "models": response.json()}
        else:
            return {"status": "Proxy server is running", "ollama_connection": "Failed", "error": response.text}
    except Exception as e:
        logger.exception(f"Error connecting to Ollama: {e}")
        return {"status": "Proxy server is running", "ollama_connection": "Failed", "error": str(e)}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        data = await request.json()
        logger.debug(f"Received request: {data}")
        
        # Convert OpenAI format to Ollama format
        ollama_data = {
            "model": data.get("model", "deepseek-r1:32b"),
            "messages": data.get("messages", []),
        }
        
        # Fix model name by removing provider prefix if present
        if '/' in ollama_data["model"]:
            ollama_data["model"] = ollama_data["model"].split('/')[-1]
            
        logger.debug(f"Sending to Ollama at {OLLAMA_URL}/api/chat: {ollama_data}")
        
        # Forward to Ollama
        try:
            response = await client.post(f"{OLLAMA_URL}/api/chat", json=ollama_data)
            logger.debug(f"Ollama response status: {response.status_code}")
            logger.debug(f"Ollama response headers: {response.headers}")
            logger.debug(f"Ollama response content: {response.text[:200]}...")  # Log first 200 chars
        except Exception as e:
            logger.exception(f"Failed to connect to Ollama: {e}")
            return Response(content=f"Failed to connect to Ollama: {str(e)}", status_code=502)
        
        if response.status_code == 200:
            try:
                ollama_response = response.json()
                logger.debug(f"Ollama response parsed successfully")
                
                # Convert Ollama response to OpenAI format
                openai_response = {
                    "id": "chatcmpl-" + str(hash(json.dumps(ollama_data)))[:10],
                    "object": "chat.completion",
                    "created": 1679351277,
                    "model": ollama_data["model"],
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": ollama_response.get("message", {}).get("content", "")
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
                
                return openai_response
            except Exception as e:
                logger.exception(f"Error processing Ollama response: {e}")
                return Response(content=f"Error processing Ollama response: {str(e)}", status_code=500)
        else:
            logger.error(f"Ollama returned error: {response.status_code}, {response.text}")
            return Response(content=response.text, status_code=response.status_code)
    except Exception as e:
        logger.exception(f"Error in proxy: {str(e)}")
        return Response(content=f"Proxy error: {str(e)}", status_code=500)

@app.get("/v1/models")
async def list_models():
    try:
        response = await client.get(f"{OLLAMA_URL}/api/tags")
        
        if response.status_code == 200:
            ollama_models = response.json().get("models", [])
            
            # Convert to OpenAI format
            openai_models = {
                "object": "list",
                "data": [
                    {
                        "id": model.get("name", "unknown"),
                        "object": "model",
                        "created": 1677610602,
                        "owned_by": "ollama"
                    }
                    for model in ollama_models
                ]
            }
            
            return openai_models
        else:
            return Response(content=response.text, status_code=response.status_code)
    except Exception as e:
        logger.exception(f"Error listing models: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info(f"Starting proxy server on port 8080, connecting to Ollama at {OLLAMA_URL}")
    uvicorn.run(app, host="0.0.0.0", port=8080)