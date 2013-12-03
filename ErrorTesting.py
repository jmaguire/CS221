from Project import Document
from collections import Counter
import numpy as np
from bleu import BLEU
import csv


class LDATester(object):
    PATH = 'DATA/'

    @staticmethod
    def compute(filename,topics = 2):
        doc = Document(LDATester.PATH + filename + '.txt')
        gold_doc = Document(LDATester.PATH + filename + '_gold.txt')
        topics = len(gold_doc.sentences)
        ldaSummary = LDATester.getSummary(doc,topics)
        # print ldaSummary
        return BLEU.computeNormalize(gold_doc.document,ldaSummary,ignore = True)
        
    @staticmethod
    def getSummary(doc,topics):
        ## Get key sentence
        sent = doc.setencesByFreqCloseness()
        maxSent = sent[0]
        # print doc.getSentenceOrginal(maxSent)

        ## Preform LDA
        doc.getLDA(topics)
        topicAndScore = doc.getTopicAndScore()
        maxTopic, maxScore = topicAndScore[maxSent]

        

        ## Create dictionary where key = topic value = sentence
        sentByTopics = {}
        for key in topicAndScore:
            value = topicAndScore[key]
            topic = value[0]
            if topic in sentByTopics:
                sentByTopics[topic] += [key]
            else:
                sentByTopics[topic] = [key]
        
        ## Statistical Cutoffs
        scores = [elem[1] for elem in topicAndScore.values()]
        mean, stdev = np.mean(scores), np.std(scores)
        maxCutoff = mean + 2*stdev
        minCutoff = mean - 1*stdev
       
        ## Collect Popular LDA sentences
        popular = []
        popular.append(doc.getSentenceOrginal(maxSent))
        
        for topic in range(doc.topics):
            
            if topic not in sentByTopics: continue
            numToAdd = 1 if topic == maxTopic else 1
            scoreAndSents = [(sentence,topicAndScore[sentence][1]) for sentence in sentByTopics[topic]]
            scoreAndSents.sort(key = lambda x: x[1], reverse = True)
            if scoreAndSents[0][1] <= minCutoff: continue
            score = 0
            index = 0
            while (score >= maxCutoff or numToAdd > 0) and index < len(scoreAndSents):
                sentence, score = scoreAndSents[index]
                if doc.getSentenceOrginal(sentence) not in popular:
                    popular.append(doc.getSentenceOrginal(sentence))
                numToAdd -= 1
                index += 1
        
        return ' '.join(popular)
        
class Calibration(object):
    PATH = 'DATA/'
    @staticmethod
    def compute(filename):
        gold_doc = Document(LDATester.PATH + filename + '_gold.txt')
        calibration = Document(FrequencyTester.PATH + 'calibration.txt')
        return BLEU.computeNormalize(gold_doc.document,calibration.document)
    
    
class FrequencyTester(object):
    PATH = 'DATA/'

    @staticmethod
    def compute(filename):
        doc = Document(FrequencyTester.PATH + filename + '.txt')
        gold_doc = Document(FrequencyTester.PATH + filename + '_gold.txt')
        freqSummary = FrequencyTester.getSummary(doc,len(gold_doc.sentences))
        return BLEU.computeNormalize(gold_doc.document,freqSummary)
        
    @staticmethod
    def getSummary(doc,length):
        ## Get key sentence
        popular = doc.setencesByFreqCloseness()
        popular = popular[0:length]
        return ' '.join(popular)

if __name__ == "__main__":
    output = [['Filename','BLEU Score LDA', 'BLEU Score Frequency', 'Calibration']]
    for i in range(10):
        filename = 'economist' + str(i + 1)
        print filename
        ldaBleu = LDATester.compute(filename,topics = 2)
        freqBleu = FrequencyTester.compute(filename)
        calibration = Calibration.compute(filename)
        line = [filename, ldaBleu, freqBleu, calibration]
        output.append(line)
    with open('Data/BLEU_Score.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(output)