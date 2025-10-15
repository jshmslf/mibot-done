from assistant import ChatBotAssistant
import random

if __name__ == "__main__":
    assistant = ChatBotAssistant('intents_en.json')
    assistant.parse_intents()
    assistant.load_model('my_chatbot_model_en.pth', 'dimensions_en.json')
    
    print("Chatbot Ready! Type /quit to exit.\n")
    while True:
        message = input("You: ")
        if message.lower() == "/quit":
            break
        print(f"jshmslf: ", assistant.process_message(message))