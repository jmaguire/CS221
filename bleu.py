from __future__ import division
from nltk.util import ngrams
import math
import nltk
## https://github.com/HengfengLi/Higher-IBM-Models/blob/master/ibm3_bleu/bleu.py
class BLEU(object):
    """
    This class implements the BLEU method, which is used to evaluate
    the quality of machine translation.

    The BLEU method mainly consists of two parts:

    Part 1 - modified n-gram precision

    The normal precision method may lead to some wrong translations with
    high-precision, e.g., the translation, in which a word of reference 
    repeats several times, has very high precision. So in the modified 
    n-gram precision, a reference word will be considered exhausted after
    a matching candidate word is identified. 

    Part 2 - brevity penalty

    As the modified n-gram precision stil has the problem from the short 
    length sentence, brevity penalty is used to modify the overall BLEU
    score according to length.

    1. Test with an instance:

    >>> weights = [0.25, 0.25, 0.25, 0.25]
    >>> candidate1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
    ...                                  'ensures', 'that', 'the', 'military', 'always', 
    ...                                  'obeys', 'the', 'commands', 'of', 'the', 'party', '.']

    >>> candidate2 = ['It', 'is', 'to', 'insure', 'the', 'troops',  
    ...               'forever', 'hearing', 'the', 'activity', 'guidebook', 
    ...               'that', 'party', 'direct', '.']

    >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 
    ...               'ensures', 'that', 'the', 'military', 'will', 'forever', 
    ...               'heed', 'Party', 'commands', '.'] 

    >>> reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',  
    ...               'guarantees', 'the', 'military', 'forces', 'always', 
    ...               'being', 'under', 'the', 'command', 'of', 'the', 
    ...               'Party', '.']

    >>> reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the', 
    ...     'army', 'always', 'to', 'heed', 'the', 'directions', 
    ...     'of', 'the', 'party', '.']

    >>> BLEU.compute(candidate1, [reference1, reference2, reference3], weights)
    0.0555662774619807

    >>> BLEU.compute(candidate2, [reference1, reference2, reference3], weights)
    0.04211415110983725

    2. Test with two corpus that one is a reference and another is 
    an output from translation system:

    >>> weights = [0.25, 0.25, 0.25, 0.25]
    >>> ref_file = open('newstest2012-ref.en')
    >>> candidate_file = open('newstest2012.fr-en.cmu-avenue')

    >>> total = 0.0
    >>> count = 0

    >>> for candi_raw in candidate_file:
    ...                ref_raw = ref_file.readline()
    ...                ref_tokens = nltk.word_tokenize(ref_raw)
    ...                candi_tokens = nltk.word_tokenize(candi_raw)
    ...                total = BLEU.compute(candi_tokens, [ref_tokens], weights)
    ...                count += 1

    >>> total/count
    2.787504437460048e-05

    """
    @staticmethod
    def compute(candidate, reference, n = 4,ignore = False):
        candidate = nltk.word_tokenize(candidate)
        reference = nltk.word_tokenize(reference)
        
        
        candidate = map(lambda x: x.lower(), candidate) 
        reference = map(lambda x: x.lower(), reference) 
        
        bp = BLEU.brevity_penalty(candidate, reference)
        if ignore:
            bp = 1
        s = 0.0
        i = 1
        for i in xrange(1,n+1,1): 
            weight = 1.0/i
            p_n = BLEU.modified_precision(candidate, reference, i)
            if p_n != 0:
                s += weight * math.log(p_n)
        return bp * math.exp(s)
    
    @staticmethod
    def computeNormalize(candidate, reference, n = 4, ignore = False):
        return BLEU.compute(candidate,reference,n,ignore)/BLEU.compute(candidate,candidate,n,ignore)

        
    @staticmethod
    def modified_precision(candidate, reference, n):

        candidate_ngrams = ngrams(candidate, n)

        if len(candidate_ngrams) == 0:
            return 0

        c_words = set(candidate_ngrams)

        for word in c_words:
            count_w = candidate_ngrams.count(word) + 1

            count_max = 0
            
            reference_ngrams = ngrams(reference, n)

            count = reference_ngrams.count(word) + 1
            if count > count_max:
                count_max = count
        return min(count_w, count_max) / (len(candidate) + len(c_words))

    @staticmethod
    def brevity_penalty(candidate, reference):
        c = len(candidate)
        
        r = len(reference)

        if c > r:
            return 1
        else:
            return math.exp(1 - r/c)

# run doctests
if __name__ == "__main__":
    sent1 = 'I went to the park and saw a small dog. He ate his mother.' 
    sent2 = 'Chicken Noodle Soup' 
    print BLEU.computeNormalize(sent1,sent2)
    