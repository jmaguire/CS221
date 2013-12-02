import nltk
from nltk.corpus import stopwords
from nltk.probability import ConditionalFreqDist
from nltk.probability import FreqDist
from gensim import corpora, models, similarities
from gensim.models import hdpmodel, ldamodel
from itertools import izip
from collections import defaultdict
import string
import numpy


class Document:
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    wordFreqKey = 'WORDFREQ'
    
    def __init__(self, path, cleanUp = True):
        rawText = open(path).read()
        self.document = rawText.decode("ascii","ignore")
        self.documentOriginal = self.document ## without stop words. So we can piece back together stuff
        
        if cleanUp: 
            self.document = self.removeStopWords(self.document)
        
        self.sentencesOriginal = Document.sent_detector.tokenize(self.documentOriginal.strip())
        
        self.paragraphs = [x.decode("ascii","ignore") for x in rawText.split('\n') if len(x) > 0]
        self.paragraphs = [Document.sent_detector.tokenize(paragraph.strip()) for paragraph in self.paragraphs]
        
        self.sentences = Document.sent_detector.tokenize(self.document.strip())
        self.cfdistPrev = self.getConditionalWordFrequenciesPrev(self.sentences)
        self.cfdistNext = self.getConditionalWordFrequenciesNext(self.sentences)
        self.freq_dist = self.getWordFrequencies(self.sentences);   
    
    
    ## --------------------------------------
    ## Frequency Closeness Measures
    ## --------------------------------------
    ## --------------------------------------
    
    def setencesByFreqCloseness(self):
        sentences = self.sentences
        # sentences.sort(key =  lambda sentence: self.freqDistributionDistance(self.freq_dist,self.getWordFrequencies([sentence])));
        sentences = [(sentence, self.freqDistributionDistance(self.freq_dist,self.getWordFrequencies([sentence]))) for sentence in self.sentences]
        sentences.sort(key = lambda x: x[1])
        sentences = [elem[0] for elem in sentences]
        return sentences

    ## Finds L1 distance between two nltk frequency distributions
    def freqDistributionDistance(self,distribution1,distribution2):
        samples = set(distribution1.samples())
        samples.union(set(distribution2.samples()))
        distance = 0
        for sample in samples:
            distance += abs(distribution1.freq(sample) - distribution2.freq(sample))
        return distance

    ## Sorts sentences by closeness to topic distributions
    def setencesByLDAFreqCloseness(self,topic):
        sentences = self.sentences
        # sentences.sort(key =  lambda sentence: self.freqDistributionDistance(self.freq_dist,self.getWordFrequencies([sentence])));
        sentences = [(sentence, self.freqLDADistributionDistance(self.ldaTopicDistributions[topic],self.getWordFrequencies([sentence]))) for sentence in self.sentences]
        sentences.sort(key = lambda x: x[1])
        sentences = [elem[0] for elem in sentences]
        return sentences
    
    ## Finds L1 distance between the document ldaDistribution and the distribution2
    def freqLDADistributionDistance(self,ldaDistribution,distribution2):
        samples = set(distribution2.samples())
        distance = 0
        for sample in samples:
            distance += abs(ldaDistribution[sample.lower()] - distribution2.freq(sample))
        return distance
    
    ## --------------------------------------
    ## --------------------------------------
    
    ## Get LDA for document
    def getLDA(self,topics):
        self.topics = topics
        texts = [[word for word in sentence.lower().split()] for sentence in self.sentences]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        self.lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=self.topics,passes=20)
        self.ldaTopicDistributions = []
        #for i in self.lda.show_topics(topn=len(dictionary)):
        for i in self.lda.show_topics():
            vector = [elem.strip() for elem in i.split('+')]
            vector = [(elem.split('*')[1],float(elem.split('*')[0])) for elem in vector]
            d = defaultdict(lambda: 0)
            for elem in vector:
                d[elem[0]] = elem[1]
            self.ldaTopicDistributions.append(d)
            
    
   
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
    
    
    ## --------------------------------------
    ## Parameterization Helper Functions
    ## --------------------------------------
    ## --------------------------------------
    
    ## returns bins of word freq closeness percentiles 10%... 100%
    def getWordFreqBins(self):
        sentences = self.sentences
        values = numpy.array([self.freqDistributionDistance(self.freq_dist,self.getWordFrequencies([sentence])) for sentence in self.sentences])
        bins = []
        for percentile in xrange(10,110,10): ##10 -> 100
            bins.append(numpy.percentile(values,percentile))
        return bins
    
    ## returns bins of length percentiles 10%... 100%
    def getLengthBins(self):
        sentences = self.sentences
        values = numpy.array([len(sentence.split()) for sentence in self.sentences])
        bins = []
        for percentile in xrange(10,110,10): ##10 -> 100
            bins.append(numpy.percentile(values,percentile))
        return bins
        
    ## used to get histogram bin value 
    def getBin(self,value, bins):
        for bin in bins:
            if value < bin: return bins.index(bin)
        if value == bins[-1]: return len(bins) - 1
        print 'this is bad'
        return None
    
    ## Returns tuple (paraNo,relativeIndex in paragraph)
    def getParagraphLocation(self,sentence):
        sentence = self.getSentenceOrginal(sentence)
        for paraNo, paragraph in enumerate(self.paragraphs):
            if sentence in paragraph:
                return paraNo, (1.0*paragraph.index(sentence)+1)/len(paragraph)
        return None,None
    ## --------------------------------------
    ## --------------------------------------
    
    def parameterize(self,sentence):
        vector = []
    
        ## Word Freq closeness
        bins = self.getWordFreqBins()
        value = self.freqDistributionDistance(self.freq_dist,self.getWordFrequencies([sentence]))
        bin = self.getBin(value, bins)
        tmp = [0]*len(bins)
        tmp[bin] = 1
        vector += tmp
        
        ## Length 
        bins = self.getLengthBins()
        value = len(sentence.split())
        bin = self.getBin(value, bins)
        tmp = [0]*len(bins)
        tmp[bin] = 1
        vector += tmp
        
        ## Location in Paragraph 
        bins = [.25,.5,.75,1]
        paraNo, value = self.getParagraphLocation(sentence)
        bin = self.getBin(value, bins)
        tmp = [0]*len(bins)
        tmp[bin] = 1
        vector += tmp
        return vector

    
    
        

















