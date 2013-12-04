from Project import Document
from collections import Counter
import numpy as np
from bleu import BLEU
import csv
import random,time,datetime

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
        doc = Document(LDATester.PATH + filename + '.txt')
        
        ## Get random summary
        indices = [x for x in range(len(doc.sentences))]
        random.shuffle(indices)
        indices = indices[0:len(gold_doc.sentences)] 
        sentences = [doc.sentences[i] for i in indices] 
        calibration = [doc.getSentenceOrginal(sentence) for sentence in sentences]
        calibration = ' '.join(calibration)
        return BLEU.computeNormalize(gold_doc.document,calibration)
    
    
class FrequencyTester(object):
    PATH = 'DATA/'

    @staticmethod
    def compute(filename):
        doc = Document(FrequencyTester.PATH + filename + '.txt')
        gold_doc = Document(FrequencyTester.PATH + filename + '_gold.txt')
        freqSummary = FrequencyTester.getSummary(doc,len(gold_doc.sentences))
        return BLEU.computeNormalize(gold_doc.document,freqSummary,ignore = True)
        
    @staticmethod
    def getSummary(doc,length):
        ## Get key sentence
        popular = doc.setencesByFreqCloseness()
        popular = popular[0:length]
        popular = [doc.getSentenceOrginal(sentence) for sentence in popular]
        return ' '.join(popular)

if __name__ == "__main__":
    title = ['Filename','BLEU Score LDA', 'BLEU Score Frequency', 'Calibration']
    results = {}
    iterations = 5
    documents = 17
    for j in range(iterations):
        print 'iteration', j
        for i in range(documents):
            filename = 'economist' + str(i + 1)
            print filename
            ldaBleu = LDATester.compute(filename,topics = 2)
            freqBleu = FrequencyTester.compute(filename)
            calibration = Calibration.compute(filename)
            if filename not in results:
                results[filename] = np.array([ldaBleu,freqBleu,calibration])
            else:
                results[filename]  += np.array([ldaBleu,freqBleu,calibration])

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%H_%M_%S')
    with open('Data/BLEU_Score_' + st + '.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(title)
        for key in results:
            writer.writerow([key] + list(results[key]/iterations*1.0))