from Project import Document
# file = raw_input("file name ")
# doc = Document(file + '.txt')
doc = Document('economist1.txt')


## Test Freq Distribution
print 'Frequency Test'
print
print 'freq of \'long\'', doc.freq_dist.freq('long')
print 'most common word', doc.freq_dist.max()
print 'num words', doc.freq_dist.N()


print 'Conditional Test previous'
print
## Test Conditional Frequency Distribution Previous
print 'most common word to follow Start', doc.cfdistPrev['Start'].max() ## most common word after Start
print 'most common word after',doc.freq_dist.max(),doc.cfdistPrev[doc.freq_dist.max()].max() ## most common word after long

print 'Conditional Test after'
print

## Test Conditional Frequency Distribution Next
print 'most common word to precede End', doc.cfdistNext['End'].max() ## most common word after Start
print 'most common word before',doc.freq_dist.max(),doc.cfdistNext[doc.freq_dist.max()].max() ## most common word after long

## get closest sentences to doc freq dist. WE WANT LDA DIST
sent = doc.setencesByFreqCloseness()
print '1', doc.getSentenceOrginal(sent[0])
print '2', doc.getSentenceOrginal(sent[1])
print '3', doc.getSentenceOrginal(sent[2])
print '4', doc.getSentenceOrginal(sent[3])
print '5', doc.getSentenceOrginal(sent[4])

with open('output.txt', 'w') as file:
    for i in [0,1,2,3,4]:
        file.write(doc.getSentenceOrginal(sent[i]) + ' ')
print ' '


## LDA
doc.getLDA()

## get MAP sentences by lda topic 1
popular_sentences = {}
for i in range(10):
    sentences = doc.setencesByLDAFreqCloseness(i)
    for j in range(5):
        sentence = doc.getSentenceOrginal(sentences[j])
        if sentence in popular_sentences:
            popular_sentences[sentence] += 1
        else:
            popular_sentences[sentence] = 1
           
popular = sorted(popular_sentences, key=popular_sentences.get, reverse = True)
print popular[0]
print popular[1]
print popular[2]
print popular[3]
print popular[4]

with open('outputLDA.txt', 'w') as file:
    for i in range(6):
        file.write(popular[i] + ' ')

