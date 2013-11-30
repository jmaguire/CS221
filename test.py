from Project import Document
file = raw_input("file name ")
doc = Document(file + '.txt')

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
