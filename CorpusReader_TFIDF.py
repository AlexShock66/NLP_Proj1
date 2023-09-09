import numpy as np
import nltk
from math import log2 as log
from nltk.corpus import inaugural

class CorpusReader_TFIDF():
    def __init__(self,corpus,tf ="raw", idf = "base", stopWord = None, toStem = False, stemFirst = False, ignoreCase = True):
        for k,v in locals().items():
            setattr(self,k,v)
    def fileids(self):
        return self.corpus.fileids()
    def raw(self,fileids=None):
        if not fileids:
            return self.corpus.raw()
        return self.corpus.raw(fileids)
    def words(self,fileids=None):
        
        if not fileids:
            _words = self.corpus.words()
        else:
            _words = self.corpus.words(fileids)
        if self.ignoreCase:
            _words = [val.lower() for val in _words]
        if self.toStem:
            stemmer = nltk.SnowballStemmer('english')
            _words = [stemmer.stem(val) for val in _words]
        return _words
    def tfidf(self,fileid,returnZero=False):
        pass
    def populate_tf(self):
        self.tf_dict = {}
        for nm in self.corpus.fileids():
            if self.tf == 'raw':
                unique_values, counts = np.unique(self.words(nm), return_counts=True)
                self.tf_dict[nm] = dict(zip(unique_values,counts))
                self.tf_dict['word_count'] = np.sum(counts)
            elif self.tf == 'log':
                unique_values, counts = np.unique(self.words(nm), return_counts=True)
                counts = [1 + log(val) for val in counts]
                self.tf_dict[nm] = dict(zip(unique_values,counts))
                self.tf_dict['word_count'] = np.sum(counts)
            else:
                raise RuntimeError(f'{self.tf} is not a valid option for tf, use either \'log\' or \'raw\'')

        
    def populate_idf(self):
        pass



if __name__ == '__main__':
    tmp = CorpusReader_TFIDF(inaugural,toStem=True)
    # tmp.populate_tf()
    print(tmp.words())
    # a = np.array(list(inaugural.words()))
    # print(tmp.tf.keys())
    # print(np.unique(a))

    # print(tmp.words(tmp.fileids()[3]))