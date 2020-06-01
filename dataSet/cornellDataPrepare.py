import ast

class CornellDataPrepare:
    def __init__(self):
        self.lines = {}
        self.conversations = []

        movieLineFields = ["lineID","characterID","movieID","character","text"]
        movieConvField = ["character1ID","character2ID","movieID","utteranceIDs"]

        self.lines = self.loadLines(movieLineFields)
        self.conversations = self.loadConversations(movieConvField)

    def loadLines(self, fields):
        lines = {}

        with open('./dataset/cornellDataset/movie_lines.txt', 'r', encoding='iso-8859-1') as file:
            for line in file:
                values = line.split(" +++$+++ ")

                lineObj = {}
                for i, field in enumerate(fields):
                    lineObj[field] = values[i]

                lines[lineObj['lineID']] = lineObj

        return lines

    def loadConversations(self, fields):
        conversations = []

        with open('./dataset/cornellDataset/movie_conversations.txt', 'r', encoding='iso-8859-1') as file:
            for line in file:
                values = line.split(" +++$+++ ")

                convObj = {}
                for i, field in enumerate(fields):
                    convObj[field] = values[i]

                lineIds = ast.literal_eval(convObj["utteranceIDs"])

                convObj["lines"] = []
                for lineId in lineIds:
                    convObj["lines"].append(self.lines[lineId])

                conversations.append(convObj)

        return conversations

    def getConversations(self):
        return self.conversations
