import numpy as np
import nltk
from math import log2 as log
from nltk.corpus import inaugural
nltk.download('inaugural')
nltk.download('stopwords')
class CorpusReader_TFIDF():
    def __init__(self,corpus,tf ="raw", idf = "base", stopWord = None, toStem = False, stemFirst = False, ignoreCase = True):
        for k,v in locals().items():
            setattr(self,k,v)
        if self.stopWord == 'none': self.stopWord = None
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
            if self.stopWord:
                if self.stemFirst:
                    _words = [stemmer.stem(val) for val in _words]
                    _words = self._handle_stop_words(_words)
                else:
                    _words = self._handle_stop_words(_words)
                    _words = [stemmer.stem(val) for val in _words]
            else:
                _words = [stemmer.stem(val) for val in _words]
        else:
            if self.stopWord:
                _words = self._handle_stop_words(_words)
                        
        return _words
    def tfidf(self,fileid,returnZero=False):
        pass

    def _populate_tf(self):
        self.tf_dict = {}
        for nm in self.corpus.fileids():
            if self.tf == 'raw':
                unique_values, counts = np.unique(self.words(nm), return_counts=True)
                self.tf_dict[nm] = dict(zip(unique_values,counts))
                self.tf_dict[nm]['word_count'] = np.sum(counts)
            elif self.tf == 'log':
                unique_values, counts = np.unique(self.words(nm), return_counts=True)
                counts = [1 + log(val) for val in counts]
                self.tf_dict[nm] = dict(zip(unique_values,counts))
                self.tf_dict[nm]['word_count'] = np.sum(counts)
            else:
                raise RuntimeError(f'{self.tf} is not a valid option for tf, use either \'log\' or \'raw\'')

    def _handle_stop_words(self,words):
        if self.stopWord == 'standard':
            stops = set(nltk.corpus.stopwords.words('english'))
            words = [val for val in words if val not in stops]
            return words
        elif self.stopWord is not None:
            with open(self.stopWord,'r') as f:
                if self.ignoreCase:
                    stops = set(f.read().lower().split('\n'))
                else:
                    stops = set(f.read().split('\n'))

                words = [word for word in words if word not in stops]
            return words
        
    def _populate_idf(self):
        if not hasattr(self,'tf_dict'):
            self._populate_tf() # Lazy load the tf dict

        self.idf_dict = {}
        tokens = set(self.words())
        for token in tokens:
            cnt = 0
            for nm in self.fileids():
                if token in self.tf_dict[nm].keys():
                    cnt += 1
            self.idf_dict[token] = cnt
        if self.idf == 'base':
            pass
        elif self.idf == 'smooth':
            pass
        else:
            raise RuntimeError(f'{self.idf} is not valid. Please use either \'base\' or \'smooth\'')-





if __name__ == '__main__':
    tmp = CorpusReader_TFIDF(inaugural,toStem=True,stopWord='standard',stemFirst=False)
    words = set(tmp.words())
    stops = set(nltk.corpus.stopwords.words('english'))
    tmp._populate_idf()
    d = tmp.idf_dict
    print(d)
    # print(len(tmp.fileids()))
    