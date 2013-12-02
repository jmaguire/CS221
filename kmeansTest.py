from Project import Document
# file = raw_input("file name ")
# doc = Document(file + '.txt')
doc = Document('economist1.txt')
# for i in range(len(doc.paragraphs)):
    # print doc.paragraphs[i]
# print doc.getParagraphLocation(doc.sentences[7])
# print doc.getWordFreqBins()
# print doc.getLengthBins()
for sentence in doc.sentences:
    doc.parameterize(sentence)

