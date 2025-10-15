import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from model import ChatBotModel
import nltk
from nltk.stem import WordNetLemmatizer

class ChatBotAssistant:
    def __init__(self,  intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path
        
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        
        self.function_mappings = function_mappings
        self.X = None
        self.y = None
        
    
    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        return words

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]
        
    def parse_intents(self):
        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)
                
            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']
                    
                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))
                
            self.vocabulary = sorted(set(self.vocabulary))
        
    def prepare_data(self):
        bags, indices = [], []
        
        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)
            
            intent_index  = self.intents.index(document[1])
            
            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatBotModel(self.X.shape[1], len(self.intents))
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss / len(loader):.4f}")
    
    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)

        with open(dimensions_path, 'w') as f:
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents)}, f)
            
    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            meta = json.load(f)
            
        # self.vocabulary = meta['vocabulary']
        # self.intents = meta['intents']
        self.model = ChatBotModel(meta['input_size'], meta['output_size'])
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        with torch.no_grad():
            predictions = self.model(bag_tensor)
            probs = F.softmax(predictions, dim = 1)
            confidence, predicted_class = torch.max(probs, dim=1)

        if confidence < 0.6:
            return "Sorry, I didn't quite get that. Can you repeat it?"
        
        intent_tag = self.intents[predicted_class.item()]
        if self.function_mappings and intent_tag in self.function_mappings:
            self.function_mappings[intent_tag]()
            
        return random.choice(self.intents_responses[intent_tag])