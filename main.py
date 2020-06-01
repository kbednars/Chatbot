import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from chatbot import Chatbot

if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot.main()

