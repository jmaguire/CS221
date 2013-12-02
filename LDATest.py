from Project import Document
from collections import Counter
import numpy as np
# filename = raw_input("file name ")
filename = 'economist1'
doc = Document(filename + '.txt')

## Get key sentence
sent = doc.setencesByFreqCloseness()
maxSent = sent[0]
print doc.getSentenceOrginal(maxSent)



doc.getLDA(5)
topicAndScore = doc.getTopicAndScore()
maxTopic, maxScore = topicAndScore[maxSent]
print topicAndScore[maxSent]

sentByTopics = {}

for key in topicAndScore:
    
    value = topicAndScore[key]
    topic = value[0]
    if topic in sentByTopics:

        sentByTopics[topic] += [key]
    else:

        sentByTopics[topic] = [key]

scores = [elem[1] for elem in topicAndScore.values()]
mean, stdev = np.mean(scores), np.std(scores)
maxCutoff = mean + 1.5*stdev
minCutoff = mean - 1*stdev
print minCutoff
popular = []
# for topic in range(doc.topics):
    # for sentence in sentByTopics[topic]:
        # print topic, sentence
# print 'HERE',sentByTopics[4]

for topic in range(doc.topics):
    if topic not in sentByTopics: continue
    numToAdd = 3 if topic == maxTopic else 1
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
print
for sent in popular:
    print sent
    
print
print

for topic in [maxTopic]:
    scoreAndSents = [(sentence,topicAndScore[sentence][1]) for sentence in sentByTopics[topic]]
    scoreAndSents.sort(key = lambda x: x[1], reverse = True)
    for i in range(3):
        print scoreAndSents[i][0]
    



# for topic in range(doc.topics):
    # print doc.freqLDADistributionDistanceCounter(doc.ldaTopicDistributions[topic],c)
# with open(str(filename) + '_LDA_output.txt', 'w') as file:
    # for i in range(len(popular)):
        # file.write(popular[i] + ' ')
# print ' '
