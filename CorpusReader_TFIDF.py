import numpy as np
import nltk
import string
from math import log2 as log
from nltk.corpus import inaugural
from collections import defaultdict

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
        return self.pre_process_word_list(_words)
        

    def pre_process_word_list(self,_words):
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
        if not hasattr(self,'tf_dict') or not hasattr(self,'idf_dict'):
            self._populate_idf() # Lazy load the tf dict
        wordz = self.words()
        if returnZero:
            dct = defaultdict(lambda:0)
        else:
            dct = {}
        for w in np.unique(wordz):
            if w in string.punctuation: continue
            val = self.tf_dict[fileid][w] * self.idf_dict[w]
            if returnZero:
                dct[w] = val
            else:
                if val != 0:
                    dct[w] = val
        return dct

    def _populate_tf(self):
        self.tf_dict = defaultdict(lambda: "Invalid Document FileId")
        for nm in self.corpus.fileids():
            if self.tf == 'raw':
                unique_values, counts = np.unique(self.words(nm), return_counts=True)
                self.tf_dict[nm] = defaultdict(lambda:0,zip(unique_values,counts))
                
            elif self.tf == 'log':
                unique_values, counts = np.unique(self.words(nm), return_counts=True)
                counts = [float(1 + log(val)) for val in counts]
                self.tf_dict[nm] = defaultdict(lambda:0,zip(unique_values,counts))
                
            else:
                raise RuntimeError(f'{self.tf} is not a valid option for tf, use either \'log\' or \'raw\'')

    def _handle_stop_words(self,words):
        stemmer = nltk.SnowballStemmer('english')
        if self.stopWord == 'standard':
            stops = set(nltk.corpus.stopwords.words('english'))
            if self.toStem:
                stops = {stemmer.stem(val) for val in stops}

            words = [val for val in words if val not in stops]
            return words
        elif self.stopWord is not None:
            with open(self.stopWord,'r') as f:
                if self.ignoreCase:
                    stops = set(f.read().lower().split('\n'))
                else:
                    stops = set(f.read().split('\n'))
                if self.toStem:
                    stops = {stemmer.stem(val) for val in stops}
                words = [word for word in words if word not in stops]
            return words
        
    def _populate_idf(self):
        if not hasattr(self,'tf_dict'):
            self._populate_tf() # Lazy load the tf dict

        self.idf_dict = defaultdict(lambda: 0) # Default dict will call lamdba function for any enntry it has not seen
        tokens = set(self.words())
        for token in tokens:
            cnt = 0
            for nm in self.fileids():
                if token in self.tf_dict[nm].keys():
                    cnt += 1
            self.idf_dict[token] = float(cnt)
        if self.idf == 'base':
            num_docs = len(self.fileids())
            for val in self.idf_dict:
                self.idf_dict[val] = log(num_docs / self.idf_dict[val])
            
        elif self.idf == 'smooth':
            num_docs = len(self.fileids())
            for val in self.idf_dict:
                self.idf_dict[val] = log( 1 + (num_docs / self.idf_dict[val]))
        else:
            raise RuntimeError(f'{self.idf} is not valid. Please use either \'base\' or \'smooth\'')


    def tfidfAll(self,returnZero=False):
        dct = defaultdict(lambda:"Invalid FileId requested")
        for id in self.fileids():
            dct[id] = self.tfidf(fileid=id,returnZero=returnZero)
        return dct
    
    def tfidfNew(self,words,returnZero=False):
        if not hasattr(self,'idf_dict'):
            self._populate_idf() # Lazy load the tf dict

        wordz = self.pre_process_word_list(words)
        unique_values, counts = np.unique(wordz, return_counts=True)
        new_tf = defaultdict(lambda:0,zip(unique_values,counts))

        wordz = np.concatenate((wordz, np.unique(self.words())))
        wordz = np.unique(wordz)


        if returnZero:
            dct = defaultdict(lambda:0)
        else:
            dct = {}
        for w in np.unique(wordz):
            if w in string.punctuation: continue
            val = new_tf[w] * self.idf_dict[w]
            if returnZero:
                dct[w] = val
            else:
                if val != 0:
                    dct[w] = val
        return dct
    def _cosine_sim_helper(self,w1,w2):
        
        #Debug:
        # w1 = {'h':3,'a':2,'b':0,'s':5}
        # w2 = {'h':1,'a':0,'b':0,'s':0}
        dot = 0
        for k in w1:
            if k not in w2.keys(): raise RuntimeError(f"Keys do not match up. Couldnt find {k}")
            dot += w1[k] * w2[k]
        w1_mag = np.sqrt(np.sum([np.power(val, 2) for val in w1.values()]))
        w2_mag = np.sqrt(np.sum([np.power(val, 2)  for val in w2.values()]))
        return dot / (w1_mag * w2_mag)
    
    def cosine_sim(self,fileid1,fileid2):
        w1 = self.tfidf(fileid=fileid1,returnZero=True)
        w2 = self.tfidf(fileid=fileid2,returnZero=True)
        return self._cosine_sim_helper(w1,w2)
    
    def cosine_sim_new(self,words,fileid):
        w1 = self.tfidfNew(words,returnZero=True)


        w2 = self.tfidf(fileid=fileid,returnZero=True)
        for word in w1.keys():
            if word not in w2.keys():
                w2[word] = 0
        
        # return w1,w2
        return self._cosine_sim_helper(w1,w2)
    def query(self,words):
        results = []
        for id in self.fileids():
            results.append((id,self.cosine_sim_new(words,id)))
        results.sort(reverse=True,key=lambda x: x[1]) 
        return results
if __name__ == '__main__':
    custom_corpus = nltk.corpus.reader.plaintext.PlaintextCorpusReader('./example','.*')
    # print(custom_corpus.words())
    mine = CorpusReader_TFIDF(custom_corpus,tf='log',idf='base',toStem=False)
    
    mine._populate_idf()
    # print(mine.tf_dict)
    # print(mine.tfidfNew(['do','be','do','run','name','be','be','think','think'],returnZero=False))
    r = mine.query(['To','be', 'or','Hello','World' ,'be','.' ,'I', 'am', 'what', 'I' ,'am','.'])
    # print(l)
    print(r)
   
    # print(mine.idf_dict)
    # print(',' in string.punctuation)
    # print(mine.tfidfAll(returnZero=True)['d4.txt'])
    # print(mine.tfidf('d1.txt'))
    # tmp = CorpusReader_TFIDF(inaugural,toStem=True,stopWord='standard',stemFirst=False,tf='log')
    # tmp._populate_tf()
    # tmp._populate_idf()
    # print(tmp.tfidf(fileid=tmp.fileids()[0]))
    # print(tmp.tf_dict.keys())
    # print(len(tmp.fileids()))
    