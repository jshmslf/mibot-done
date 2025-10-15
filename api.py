from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from assistant import ChatBotAssistant

app = FastAPI()

# frontend request
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
assistant = ChatBotAssistant('intents_en.json')
assistant.parse_intents()
assistant.load_model('my_chatbot_model_en.pth', 'dimensions_en.json')

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message", "")
    if not message.strip():
        return {"response": "Please say something"}
    
    response = assistant.process_message(message)
    return {"responses": response}