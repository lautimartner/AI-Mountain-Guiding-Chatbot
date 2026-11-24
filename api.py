# FastAPI is the web framework: it turns Python functions into HTTP endpoints.
from fastapi import FastAPI, HTTPException

# CORSMiddleware lets the browser call this API from a different website.
from fastapi.middleware.cors import CORSMiddleware

# BaseModel defines the shape of JSON data; Field adds rules like min and max length.
from pydantic import BaseModel, Field

# The answer() function from ask.py: it retrieves route chunks and asks Gemini.
from ask import answer

# Create the application object. Uvicorn looks for this variable when it runs "api:app".
app = FastAPI()

# Add the CORS rules as middleware, so every response includes the headers the browser needs.
app.add_middleware(
    # The class that handles the CORS checks.
    CORSMiddleware,
    # Only these websites may call the API from a browser. Add the Netlify URL while testing.
    allow_origins=["http://localhost:5500", "https://sichuanmountains.org"],
    # Only POST requests are allowed from other sites, which is all the chat needs.
    allow_methods=["POST"],
    # Only the Content-Type header is allowed, since the widget sends JSON.
    allow_headers=["Content-Type"],
)


# Describes the JSON the browser must send: a "question" field of 1 to 500 characters.
class ChatRequest(BaseModel):
    # The customer's question. FastAPI rejects the request with a 422 error if the rules fail.
    question: str = Field(min_length=1, max_length=500)


# Describes the JSON the API returns: an "answer" field containing the text.
class ChatResponse(BaseModel):
    # The chatbot's reply, as plain text.
    answer: str


# Register this function for POST requests to /api/chat, and return data matching ChatResponse.
@app.post("/api/chat", response_model=ChatResponse)
# FastAPI reads the request body, validates it as ChatRequest, and passes it in as req.
def chat(req: ChatRequest):
    # Start a try block, so any unexpected error is caught below instead of crashing the server.
    try:
        # Remove extra spaces from the question, then run the retrieval and Gemini call.
        # The result is wrapped in ChatResponse, which FastAPI converts to JSON.
        return ChatResponse(answer=answer(req.question.strip()))
    # Handle any error raised inside the try block.
    except Exception as e:
        # Print the real error to the terminal, so you can debug it. The customer doesn't see this.
        print("Error:", e)
        # Send an HTTP 500 response with a generic message, so internal details stay private.
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")


# Register a simple GET endpoint that checks whether the server is running.
@app.get("/api/health")
# Return a small JSON object. Visiting this URL in a browser should show {"ok": true}.
def health():
    # Send back a dictionary, which FastAPI converts to JSON.
    return {"ok": True}