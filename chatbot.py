import numpy as np
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import yaml
from keras.layers import Input, Embedding, GlobalMaxPool1D
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense
from textProcess import TextProcess

class Chatbot:
    def __init__(self):
        self.parameters = None
        self.textProcess = None
        self.model = None

    def main(self):
        print('Welcome to Chatbot')
        stream = open("parameters.txt", 'r')
        self.parameters = yaml.safe_load(stream)

        self.textProcess = TextProcess(self.parameters)

        if not os.path.isfile('chatbotModel.pkl'):
            input_context = Input(shape=(self.parameters['maxLength'],), dtype="int32", name="input_context")
            embeddingLayer =  Embedding(self.parameters['vocabularySize'],
                                output_dim=self.parameters['embeddingOutputDim'],
                                weights=[self.textProcess.embedding_matrix],
                                input_length=self.parameters['maxLength'],
                                trainable=True)
            embedding_context = embeddingLayer(input_context)

            layer = GlobalMaxPool1D()(embedding_context)
            layer = Dense(int(self.parameters['trainData']/2), activation='relu')(layer)
            outputs = Dense(self.parameters['trainData'], activation='softmax')(layer)

            self.model = Model(input=[input_context], output=[outputs])
            adam = Adam(lr=self.parameters['learningRate'])
            self.model.compile(loss="categorical_crossentropy", optimizer=adam)
            self.model.summary()
            self.model.fit(self.textProcess.inputSequences[:self.parameters['trainData']],
                           self.textProcess.getDecoderOutputData(self.textProcess.targetSequences[:self.parameters['trainData']]),
                           batch_size=self.parameters['batchSize'],
                           epochs=self.parameters['numEpochs'])
            self.saveDataset()
        else:
            self.loadDataset()
        self.start_chatbot()

    def start_chatbot(self):
        while True:
            question = input('You: ')
            if question == '' or question == 'exit':
                break
            answer = self.decode_sequence(question)

            print('Bot: ' + format(answer))
            print()

    def decode_sequence(self, input_seq):
        target_seq = self.textProcess.getSentenceTokens(input_seq)
        states_value = self.model.predict(target_seq)
        sampled_token_index = np.argmax(states_value[0, :])
        decoded_sentence = self.textProcess.targetData[sampled_token_index]
        return decoded_sentence

    def saveDataset(self):
        with open('chatbotModel.pkl', 'wb') as handle:
            data = {'model': self.model}
            pickle.dump(data, handle, -1)

    def loadDataset(self):
        with open('chatbotModel.pkl', 'rb') as handle:
            data = pickle.load(handle)
            self.model = data['model']