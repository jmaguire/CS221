import nltk
from nltk.corpus import stopwords
from nltk.probability import ConditionalFreqDist
from nltk.probability import FreqDist
import string

class Document:
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    wordFreqKey = 'WORDFREQ'
    
    def __init__(self, path, cleanUp = True):
        self.document = open(path).read()
        self.documentOriginal = self.document ## without stop words. So we can piece back together stuff
        
        if cleanUp: 
            self.document = self.removeStopWords(self.document)
        
        self.sentencesOriginal = Document.sent_detector.tokenize(self.documentOriginal.strip())
        
        self.sentences = Document.sent_detector.tokenize(self.document.strip())
        self.cfdistPrev = self.getConditionalWordFrequenciesPrev(self.sentences)
        self.cfdistNext = self.getConditionalWordFrequenciesNext(self.sentences)
        self.freq_dist = self.getWordFrequencies(self.sentences);   
        
    
    def freqDistDistance(self,distribution1,distribution2):
        samples = set(distribution1.samples())
        samples.union(set(distribution2.samples()))
        distance = 0
        for sample in samples:
            distance += abs(distribution1.freq(sample) - distribution2.freq(sample))
        return distance
    
    
    def setencesByFreqCloseness(self):
        sentences = self.sentences
        # sentences.sort(key =  lambda sentence: self.freqDistDistance(self.freq_dist,self.getWordFrequencies([sentence])));
        sentences = [(sentence, self.freqDistDistance(self.freq_dist,self.getWordFrequencies([sentence]))) for sentence in self.sentences]
        #print sentences
        sentences.sort(key = lambda x: x[1])
        sentences = [elem[0] for elem in sentences]
        return sentences
    
    ## condition on previous word    
    ## Takes list of sentences
    def getConditionalWordFrequenciesPrev(self,sentences):
        cfdist = ConditionalFreqDist()
        condition = 'Start'
        for sentence in sentences:
            for token in nltk.word_tokenize(sentence):
                if token in ['.','?','!']:
                    condition = 'Start'
                    continue
                cfdist[condition].inc(token)
                condition = token
        return cfdist
    
    ## conditional on Next Word 
    ## Takes list of sentences
    def getConditionalWordFrequenciesNext(self,sentences):
        cfdist = ConditionalFreqDist()
        condition = None
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            for index, token in enumerate(tokens):
                if token in ['.','?','!'] or index == len(tokens) - 1:
                    continue
                condition = tokens[index + 1]
                if condition in ['.','?','!']:
                    condition = 'End'
                cfdist[condition].inc(token)
        return cfdist
    
    ## Takes list of sentences. If single sentence pass [sentence] as arguement
    def getWordFrequencies(self, sentences):
        freq_dist = FreqDist()
        for sentence in sentences:
            for token in nltk.word_tokenize(sentence):
                if token not in string.punctuation:
                    freq_dist.inc(token)
        return freq_dist

    def removeStopWords(self,string):
        stopset = set(stopwords.words('english'))
        tokens = nltk.word_tokenize(string)
        cleanup = [token for token in tokens if token not in stopset and len(token)>2]
        return ' '.join(cleanup)

    def getSentenceOrginal(self,sentence):
        index = self.sentences.index(sentence)
        return self.sentencesOriginal[index]
        
    
        

















