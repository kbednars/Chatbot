import numpy as np
import nltk
from tqdm import tqdm
import pickle
import os
import re
from dataSet.cornellDataPrepare import CornellDataPrepare
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import json

class TextProcess:

    def __init__(self, parameters):
        self.parameters = parameters

        self.trainingData = []
        self.inputData = []
        self.inputSequences = []
        self.targetData = []
        self.targetSequences = []
        self.embedding_matrix = []
        self.embeddings_index = {}

        self.word2index = {}
        self.index2word = {}

        self.dataSet = CornellDataPrepare()
        self.prepareData()

    def prepareData(self):
        if not os.path.isfile('./dataSet/textProcessData.pkl'):
            file = open('dataSet/train-v2.0.json', 'r', encoding='iso-8859-1')
            parsed_json = (json.load(file))
            for i in tqdm_wrap(parsed_json['data'][:self.parameters['dataPart']], desc='Conversation', leave=False):
                for j in i['paragraphs']:
                    for k in j['qas']:
                        if k['answers'] != []:
                            inputLine = self.clean_text(k['question'])
                            targetLine = self.clean_text(k['answers'][0]['text'])
                            self.inputData += [inputLine]
                            self.targetData += [targetLine]

            self.word2index, self.index2word = self.createVocabluary(self.inputData, self.targetData)
            self.embeddingMatrix()
            self.saveDataset()
        else:
            self.loadDataset()

    def clean_text(self, line):
        line = line.lower()
        line = re.sub(r"i'm", "i am", line)
        line = re.sub(r"he's", "he is", line)
        line = re.sub(r"she's", "she is", line)
        line = re.sub(r"it's", "it is", line)
        line = re.sub(r"that's", "that is", line)
        line = re.sub(r"what's", "what is", line)
        line = re.sub(r"where's", "where is", line)
        line = re.sub(r"how's", "how is", line)
        line = re.sub(r"\'ll", " will", line)
        line = re.sub(r"\'ve", " have", line)
        line = re.sub(r"\'re", " are", line)
        line = re.sub(r"\'d", " would", line)
        line = re.sub(r"\'re", " are", line)
        line = re.sub(r"won't", "will not", line)
        line = re.sub(r"can't", "cannot", line)
        line = re.sub(r"n't", " not", line)
        line = re.sub(r"n'", "ng", line)
        line = re.sub(r"'bout", "about", line)
        line = re.sub(r"'til", "until", line)
        line = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", line)
        return line

    def createVocabluary(self, encoderData, decoderData):
        tokenizer = Tokenizer(num_words=self.parameters['vocabularySize'])
        tokenizer.fit_on_texts(encoderData+decoderData)
        print('Tokenizer fitted')
        self.inputSequences = tokenizer.texts_to_sequences(encoderData)
        self.targetSequences = tokenizer.texts_to_sequences(decoderData)
        self.inputSequences = pad_sequences(self.inputSequences, maxlen=self.parameters['maxLength'], dtype='int32', padding='post', truncating='post')
        self.targetSequences = pad_sequences(self.targetSequences, maxlen=self.parameters['maxLength'], dtype='int32', padding='post', truncating='post')

        dictionary = tokenizer.word_index

        word2index = {}
        index2word = {}
        for k, v in dictionary.items():
            if v < self.parameters['vocabularySize']:
                word2index[k] = v
                index2word[v] = k
            if v >= self.parameters['vocabularySize'] - 1:
                continue
        print('Vocabulary updated')
        return word2index, index2word

    def embeddingMatrix(self):
        self.embeddings_index = {}
        file = open('./glove.6B.100d.txt', encoding='utf8')
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        file.close()

        self.embedding_matrix = np.zeros((len(self.word2index) + 1, 100))
        for word, i in self.word2index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector

    def saveDataset(self):
        with open('./dataSet/textProcessData.pkl', 'wb') as handle:
            data = {
                'word2index': self.word2index,
                'index2word': self.index2word,
                'inputData': self.inputData,
                'targetData': self.targetData,
                'inputSeq': self.inputSequences,
                'targetSeq': self.targetSequences,
                'embedding_matrix': self.embedding_matrix
            }
            pickle.dump(data, handle, -1)

    def loadDataset(self):
        with open('./dataSet/textProcessData.pkl', 'rb') as handle:
            data = pickle.load(handle)
            self.word2index = data['word2index']
            self.index2word = data['index2word']
            self.inputData = data['inputData']
            self.targetData = data['targetData']
            self.inputSequences = data['inputSeq']
            self.targetSequences = data['targetSeq']
            self.embedding_matrix = data['embedding_matrix']

    def getDecoderOutputData(self, sequences):
        decoderOutput = np.zeros((sequences.shape[0], self.parameters['trainData']), dtype="float16")
        for i, sequence in enumerate(sequences):
            decoderOutput[i][i] = 1.
        return decoderOutput

    def getSentenceTokens(self, sentence):
        sentence = nltk.sent_tokenize(sentence)
        sentence = self.clean_text(sentence[0])
        tokenizer = Tokenizer(num_words=self.parameters['vocabularySize'])
        tokenizer.fit_on_texts(self.inputData + self.targetData)
        sentence = tokenizer.texts_to_sequences([sentence])
        sentence = pad_sequences(sentence, maxlen=self.parameters['maxLength'], dtype='int32',
                                              padding='post', truncating='post')
        return sentence

    def getSentecteFromPrediction(self, prediction):
        sentece = ''
        for i in range(self.parameters['maxLength']):
            temp = np.argmax(prediction)
            prediction[0,temp] = 0
            if temp != 1 and temp != 2:
                sentece += self.index2word[temp] + ' '
        return sentece


def tqdm_wrap(iterable, *args, **kwargs):
    if len(iterable) > 100:
        return tqdm(iterable, *args, **kwargs)
    return iterable
