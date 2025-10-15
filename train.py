from assistant import ChatBotAssistant

if __name__ == "__main__":
    assistant = ChatBotAssistant('intents_en.json')
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(batch_size=8, lr=0.001, epochs=150)
    assistant.save_model('my_chatbot_model_en.pth', 'dimensions_en.json')