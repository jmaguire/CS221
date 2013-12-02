from Project import Document
# filename = raw_input("file name ")
filename = 'economist3'
doc = Document(filename + '.txt')
# doc = Document('economist1.txt')
print doc.documentOriginal

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

with open(str(filename) + '_output.txt', 'w') as file:
    for i in [0,1,2,3,4]:
        file.write(doc.getSentenceOrginal(sent[i]) + ' ')
print ' '


## LDA

from collections import Counter
## get MAP sentences by lda topic i
popular = Counter()
for n in range(20):
    doc.getLDA(6)
    for i in range(doc.topics):
        sentences = doc.setencesByLDAFreqCloseness(i)
        
        popular.update(Counter([doc.getSentenceOrginal(sentence) for sentence in sentences[0:1]]))
        # for j in range(3):
            # sentence = doc.getSentenceOrginal(sentences[j])
            # if sentence in popular_sentences:
                # popular_sentences[sentence] += 1
            # else:
                # popular_sentences[sentence] = 1
popular = [elem[0] for elem in popular.most_common(6)]
# popular = [sent for sent in set(popular_sentences)]
# popular = sorted(popular_sentences, key=popular_sentences.get, reverse = True)
for sent in popular:
    print sent

with open(str(filename) + '_LDA_output.txt', 'w') as file:
    for i in range(len(popular)):
        file.write(popular[i] + ' ')
print ' '
