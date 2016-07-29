""" Module containing feature generators used for learning.
    I think I reinvented sklearn pipelines - too late now!
    A dictionary of functions is used for feature generation.
    If a function has only one argument feature generation is
    independent of training or test case.
    If it takes two arguments, feature generation depends
    on case - for example: bag_of_words
    This is supposed to be extensible as you can add or remove
    any functions you like from the dictionary
"""
import regex as re
import nltk
import numpy
from textblob.tokenizers import WordTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from misc import _twokenize


def cleaned(term_prof, thres = 0.1):

    import numpy

    k = 0
    for i in range(0, term_prof.shape[0]):
        tmp = list(term_prof[i, :])
        minim = min(tmp)
        for prob in tmp:
            if prob != minim:
                if abs(prob - minim) < thres:
                    term_prof[i, :] = 100 * numpy.ones([1, term_prof.shape[1]])
                    k += 1
                    break
    print("We cleared " + str(k) + "  terms! This is the " + str(100 * k / float(term_prof.shape[0])) + " percent.")
    return term_prof


def logloss(y_true, Y_pred):

    import math
    import numpy as np

    label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
    return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in zip(Y_pred, y_true)) / len(Y_pred)

def tokenization(text):

        import re
        # from nltk.stem import WordNetLemmatizer

        # Create reg expressions removals
        nonan = re.compile(r'[^a-zA-Z ]')  # basically numbers
        # po_re = re.compile(r'\.|\!|\?|\,|\:|\(|\)')  # punct point and others
        temp2 = nonan.sub('', text).lower().split()
        # temp = nonan.sub('', po_re.sub('', text)).lower().split()
        # print temp
        # temp2 = [WordNetLemmatizer().lemmatize(item, 'v') for item in temp2]
        return temp2


def tokenization2(text):

        import re

        emoticons_str = r"""
        (?:
          [:=;] # Eyes
          [oO\-]? # Nose (optional)
          [D\)\]\(\]/\\OpP] # Mouth
        )"""

        regex_str = [
            emoticons_str,
            r'<[^>]+>',  # HTML tags
            r'(?:@[\w_]+)',  # @-mentions
            r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
            r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
            r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
            r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
            r'(?:[\w_]+)',  # other words
            r'(?:\S)'  # anything else
        ]
        tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
        emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

        return [token if emoticon_re.search(token) else token.lower() for token in tokens_re.findall(text)]


# ------------------------ feature generators --------------------------------#



class TopicTopWords(BaseEstimator, TransformerMixin):

    """ Suppose texts can be split into n topics. Represent each text
        as a percentage for each topic."""

    def __init__(self, n_topics, k_top):
        from sklearn.feature_extraction.text import CountVectorizer
        self.n_topics = n_topics
        self.k_top = k_top
        self.model = LDA(n_topics=self.n_topics,
                             n_iter=10,
                             random_state=1)
        self.counter = CountVectorizer()

    def fit(self, X, y=None):
        X = self.counter.fit_transform(X)
        self.model.fit(X)
        return self

    def transform(self, texts):
        """ transform data

        :texts: The texts to count hashes in
        :returns: list of counts for each text

        """
        X = self.counter.transform(texts).toarray()  # get counts for each word
        topic_words = self.model.topic_word_  # model.components_ also works
        topics = numpy.hstack([X[:, numpy.argsort(topic_dist)]
                                [:, :-(self.k_top + 1):-1]
                               for topic_dist in topic_words])
        return topics


class PrintLen(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of twitter-style hashes. """

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        """ transform data

        :texts: The texts to count hashes in
        :returns: list of counts for each text

        """
        print(texts.shape)
        return texts


class CountHash(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of twitter-style hashes. """

    pat = re.compile(r'(?<=\s+|^)#\w+', re.UNICODE)

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        """ transform data

        :texts: The texts to count hashes in
        :returns: list of counts for each text

        """
        return [[len(CountHash.pat.findall(text)) / float(len(text))] for text in texts]


class CountReplies(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of twitter-style @replies. """

    pat = re.compile(r'(?<=\s+|^)@\w+', re.UNICODE)

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        """ transform data

        :texts: The texts to count replies in
        :returns: list of counts for each text

        """
        return [[len(CountReplies.pat.findall(text)) / float(len(text))] for text in texts]


class CountURLs(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of URL links from text. """

    pat = re.compile(r'((https?|ftp)://[^\s/$.?#].[^\s]*)')

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        """ transform data

        :texts: The texts to count URLs in
        :returns: list of counts for each text

        """
        return [[len(CountURLs.pat.findall(text)) / float(len(text))] for text in texts]


class CountCaps(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of capital letters from text. """

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        """ transform data

        :texts: The texts to count capital letters in
        :returns: list of counts for each text

        """
        return [[sum(c.isupper() for c in text)] for text in texts]


class CountWordCaps(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of capital words from text. """

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        """ transform data

        :texts: The texts to count capital words in
        :returns: list of counts for each text

        """
        return [[sum(w.isupper() for w in nltk.word_tokenize(text))]
                for text in texts]


class CountWordLength(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of word length from text. """

    def __init__(self, span):
        """ Initialize this feature extractor
        :span: tuple - range of lengths to count

        """
        self.span = span

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        """ transform data

        :texts: The texts to count word lengths in
        :returns: list of counts for each text

        """

        mini = self.span[0]
        maxi = self.span[1]
        num_counts = maxi - mini
        # wt = WordTokenizer()
        tokens = [tokenization(text) for text in texts]
        text_len_dist = []
        for line_tokens in tokens:
            counter = [0] * num_counts
            for word in line_tokens:
                word_len = len(word)
                if mini <= word_len <= maxi:
                    counter[word_len - 1] += 1
            text_len_dist.append([each for each in counter])
        return text_len_dist


class CountTokens(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of capital words from text. """

    def __init__(self):
        self.l = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                  'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                  'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3',
                  '4', '5', '6', '7', '8', '9', '!', '.', ':', '?']
                  #';', ',', ')', '(', '-', '%', '$', '#', '@', '^',
                  #'&', '*', '=', '+', '/', '"', "'", '<', '>', '|',
                  #'~', '`']

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        """ transform data

        :texts: The texts to count capital words in
        :returns: list of counts for each text

        """
        l = self.l
        return [[text.lower().count(token)/float(len(text)) for token in l]
                for text in texts]

# class SOA_Model2(object):

class SOAC_Model2(BaseEstimator, TransformerMixin):

    """ Complementary of SOA model 22"""

    def __init__(self, max_df=1.0, min_df=1,
                 tokenizer_var='sklearn', max_features=None, thres = 0.1):
        from sklearn.feature_extraction.text import TfidfVectorizer

        # stop_list = []
        # with open(stopwords_path, 'r') as stop_inp:
        # for w in stop_inp:
        # stop_list.append(w.replace("\n", ""))
        # print tokenizer_var
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.tokenizer_var = tokenizer_var
        self.thres = thres 
        self.term_table = None
        self.labels = None
        self.prior_row = None
        if self.tokenizer_var == '1':
            self.tokenization = tokenization
        elif self.tokenizer_var == '2':
            self.tokenization = tokenization2
        elif self.tokenizer_var == '3':
            self.tokenization = _twokenize.tokenizeRawTweetText
        else:
            self.tokenization = None

        # self.lsi = None
        # self.dictionary = None
        # self.num_topics = 100
        # self.counter = CountVectorizer()
        self.counter = TfidfVectorizer(use_idf=True)

    def fit(self, X, y=None):

        import numpy
        from sklearn.preprocessing import normalize
        #print "We are fitting!"
        if y is None:
            raise ValueError('we need y labels to supervise-fit!')
        else:
            parameters = {
                'input': 'content',
                'encoding': 'utf-8',
                'decode_error': 'ignore',
                'analyzer': 'word',
                'tokenizer': self.tokenization,
                # 'vocabulary':list(voc),
                # 'tokenizer': tokenization,
                #'tokenizer': _twokenize.tokenizeRawTweetText,  # self.tokenization,
                #'tokenizer': lambda text: _twokenize.tokenizeRawTweetText(nonan.sub(po_re.sub('', text))),
                'max_df': self.max_df,
                'min_df': self.min_df,
                'max_features': self.max_features
            }
            self.counter.set_params(**parameters)
            import time
            time_start = time.time()
            doc_term = self.counter.fit_transform(X)
            #print "To tf-idf completed in %0.2f sec" % (time.time() - time_start)
            target_profiles = sorted(list(set(y)))
            self.labels = target_profiles
            from collections import Counter
            dd = Counter(y)
            self.prior_row = numpy.zeros([1, len(target_profiles)])
            for i, key in enumerate(sorted(dd.keys())):
                dd[key] = dd[key]/float(len(y))
                self.prior_row[0, i] = 1/dd[key]
            #print self.prior_row
            # Gia to palio. Na vgalw kai doc_prof apo to transform
            #self.prior_row = numpy.ones([1, len(target_profiles)])
            doc_prof = numpy.tile(self.prior_row, (doc_term.shape[0], 1))
            #print "Doc_PROF1"
            #print doc_prof
            #doc_prof = numpy.ones([doc_term.shape[0], len(target_profiles)])
            for i in range(0, doc_term.shape[0]):
                doc_prof[i, target_profiles.index(y[i])] = 0
                #tmp = numpy.ones([1, len(target_profiles)])
                #tmp[0, target_profiles.index(y[i])] = 0
                #doc_prof[i, :] = tmp
            #doc_prof = doc_prof / doc_prof.sum(axis=0)
            #print "Created doc_prof matrix in %0.2f sec" % (time.time() - time_start)
            #import random
            #print "example"
            # print doc_prof[random.randint(1, 10), :]
            #import pprint
            #print "VOcab"
            #pprint.pprint(self.counter.vocabulary_)
            #print "doc_prof"
            #pprint.pprint(doc_prof)
            #print "doc_term"
            #pprint.pprint(doc_term.transpose().toarray())
            #print "Type doc_term"
            #print type(doc_term)
            #print doc_term.shape
            #print type(doc_term.data)
            #print "Type doc_prof"
            #print type(doc_prof)
            #print doc_prof.shape
            try:
                doc_term.data = numpy.log2(doc_term.data + 1)
            except Exception, e:
                print "Error in log2"
                print e
            try:
                term_prof = doc_term.transpose().dot(doc_prof)
            except Exception, e:
                print "Error in product"
                print e

            #doc_term.data = numpy.log2(doc_term.data + 1)
            #print "To tf-idf + log completed in %0.2f sec" % (time.time() - time_start)
            #term_prof = doc_term.transpose().dot(doc_prof)
            #print "Completed dot product in %0.2f sec" % (time.time()- time_start)
            #print "term_table"
            #pprint.pprint(term_prof)
            # normalize against words
            term_prof = term_prof / term_prof.sum(axis=0)
            #print "Normalization per collumn product in %0.2f sec" % (time.time()- time_start)
            #normalize(term_prof, norm='l1', axis=0, copy=False)
            # normalize across profiles
            term_prof = term_prof / \
                numpy.reshape(
                   term_prof.sum(axis=1), (term_prof.sum(axis=1).shape[0], 1))
            #print "Normalization per collumn product in %0.2f sec" % (time.time()- time_start)
            #normalize(term_prof, norm='l1', axis=1, copy=False)
            # clean term_prof
            # term_prof = cleaned(term_prof, self.thres)
            self.term_table = term_prof
            import pprint
            #print "term_table"
            #pprint.pprint(self.term_table)
            return self

    def transform(self, X, y=None):

        import numpy

        #print "We are transforming!"
        if self.labels is None:
            raise AttributeError('term_table was no found! \
             Probably model was not fitted first. Run model.fit(X,y)!')
        else:
            #print 'DIAVAZW'
            #print type(X)
            #print X
            doc_term = self.counter.transform(X)
            #doc_prof = numpy.zeros(
            #    [doc_term.shape[0], self.term_table.shape[1]])
            doc_prof = doc_term.dot(self.term_table)
            #print 'Doc_prof'
            #print doc_prof.shape, type(doc_prof)
            # fake norm
            for i in range(0, doc_prof.shape[0]):
                doc_prof[i, :] = doc_prof[i, :] #- doc_prof[i, :].min()
            # print doc_prof
            #from sklearn.preprocessing import normalize
            #normalize(doc_prof, norm='l1', axis=1, copy=False)
            return doc_prof


class SOA_Model2(BaseEstimator, TransformerMixin):

    """ Models that extracts Second Order Attributes
     (SOA) base on PAN 2013-2015 Winners asd"""

    def __init__(self, max_df=1.0, min_df=5,
                 tokenizer_var='sklearn', max_features=None):
        from sklearn.feature_extraction.text import TfidfVectorizer

        # stop_list = []
        # with open(stopwords_path, 'r') as stop_inp:
        # for w in stop_inp:
        # stop_list.append(w.replace("\n", ""))
        # print tokenizer_var
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.tokenizer_var = tokenizer_var
        self.term_table = None
        self.labels = None
        if self.tokenizer_var == '1':
            self.tokenization = tokenization
        elif self.tokenizer_var == '2':
            self.tokenization = tokenization2
        elif self.tokenizer_var == '3':
            self.tokenization = _twokenize.tokenizeRawTweetText
        else:
            self.tokenization = None

        # self.lsi = None
        # self.dictionary = None
        # self.num_topics = 100
        # self.counter = CountVectorizer()
        self.counter = TfidfVectorizer(use_idf=False)

    def fit(self, X, y=None):

        import numpy
        from sklearn.preprocessing import normalize

        #print "We are fitting!"
        if y is None:
            raise ValueError('we need y labels to supervise-fit!')
        else:
            # texts = [self.tokenization(text) for text in X]
            # self.dictionary = corpora.Dictionary(texts)
            # corpus = [self.dictionary.doc2bow(text) for text in texts]
            # self.lsi = models.LsiModel(corpus, id2word=self.dictionary, num_topics=self.num_topics)
            # for token in tokens:
            #    voc = voc.union(token)
            # print len(voc)
            # print list(voc)[:100]
            parameters = {
                'input': 'content',
                'encoding': 'utf-8',
                'decode_error': 'ignore',
                'analyzer': 'word',
                'tokenizer': self.tokenization,
                # 'vocabulary':list(voc),
                # 'tokenizer': tokenization,
                #'tokenizer': _twokenize.tokenizeRawTweetText,  # self.tokenization,
                #'tokenizer': lambda text: _twokenize.tokenizeRawTweetText(nonan.sub(po_re.sub('', text))),
                'max_df': self.max_df,
                'min_df': self.min_df,
                'max_features': self.max_features
            }
            self.counter.set_params(**parameters)
            # print str(self.counter.get_params())
            # print len(target_profiles)
            doc_term = self.counter.fit_transform(X)
            # st_scaler = StandardScaler(copy=False)
            # st_scaler.fit_transform(doc_term)
            #normalize(doc_term, norm='l1', axis=0, copy=False)
            #print "Doc_Terms"
            #print doc_term.shape
            target_profiles = sorted(list(set(y)))
            self.labels = target_profiles
            doc_prof = numpy.zeros([doc_term.shape[0], len(target_profiles)])
            for i in range(0, doc_term.shape[0]):
                tmp = numpy.zeros([1, len(target_profiles)])
                tmp[0, target_profiles.index(y[i])] = 1
                doc_prof[i, :] = tmp
            #print "Doc_Prof"
            #print doc_prof.shape, type(doc_prof)
            doc_term.data = numpy.log2(doc_term.data + 1)
            #doc_term.transpose
            #print "Doc_Term"
            #print doc_term.shape, type(doc_term)
            term_prof = doc_term.transpose().dot(doc_prof)
            #term_prof = numpy.zeros([doc_term.shape[1], len(target_profiles)])
            #term_prof = numpy.log2(doc_term.transpose.data
            #term_prof = numpy.dot(
            #    numpy.log2(doc_term.toarray().astype('float', casting='unsafe').T + 1), doc_prof)
            #print "Term_Prof"
            #print term_prof.shape, type(term_prof)
            # normalize against words
            term_prof = term_prof / term_prof.sum(axis=0)
            # normalize(term_prof, norm='l1', axis=0, copy=False)
            # normalize across profiles
            term_prof = term_prof / \
                numpy.reshape(
                   term_prof.sum(axis=1), (term_prof.sum(axis=1).shape[0], 1))
            # normalize(term_prof, norm='l1', axis=0, copy=False)
            #print "Random Term_Prof"
            #print term_prof[0,:]
            # term_prof = term_prof / \
            #    numpy.reshape(
            #        term_prof.sum(axis=1), (term_prof.sum(axis=1).shape[0], 1))
            # term_prof = term_prof / term_prof.sum(axis=0)
            self.term_table = term_prof
            #print "SOA Model Fitted!"
            return self

    def transform(self, X, y=None):

        import numpy

        #print "We are transforming!"
        if self.labels is None:
            raise AttributeError('term_table was no found! \
             Probably model was not fitted first. Run model.fit(X,y)!')
        else:
            #doc_term = numpy.zeros(
            #    [len(X), self.term_table.shape[0]])
            doc_term = self.counter.transform(X)
            #print "Doc_Terms"
            #print doc_term.shape, type(doc_term)
            doc_prof = numpy.zeros(
                [doc_term.shape[0], self.term_table.shape[1]])
            # print "Term_Prof"
            # print self.term_table.shape
            doc_prof = doc_term.dot(self.term_table)
            # doc_prof = numpy.dot(
            #    doc_term.toarray().astype('float', casting='unsafe'),
            #    self.term_table)
            #print "SOA Transform:"
            # print type(doc_prof)
            #print 'Doc_prof'
            #print doc_prof.shape, type(doc_prof)
            #print doc_prof[0,:]
            #print "Len Voc: %s" % (str(len(self.counter.vocabulary_)))
            # import pprint
            # pprint.pprint(self.counter.vocabulary_)
            # LSI
            # texts = [self.tokenization(text) for text in X]
            # corpus = [self.dictionary.doc2bow(text) for text in texts]
            # transform_lsi = self.lsi[corpus]
            # lsi_list = []
            # dummy_empty_list = [0 for i in range(0, self.num_topics)]
            # #c = 0
            # for i, doc in enumerate(transform_lsi):
            #     if not doc:  # list is empty
            #         lsi_list.append(dummy_empty_list)
            #     else:
            #         lsi_list.append(list(zip(*doc)[1]))
            #         if len(lsi_list[-1]) != self.num_topics:
            #             # c += 1
            #             # print c
            #             # print texts[i]
            #             # print len(lsi_list[-1])
            #             # print lsi_list[-1]
            #            lsi_list[-1] = dummy_empty_list
            # lsi_list = [list(zip(*doc)[1]) for doc in transform_lsi]
            # print numpy.array(lsi_list).shape
            # print len(lsi_list)
            # print len(lsi_list[0])
            # c = numpy.reshape(numpy.array(lsi_list), (len(lsi_list), self.num_topics))
            # print c.shape
            # return numpy.hstack((doc_prof, numpy.reshape(numpy.array(lsi_list), (len(lsi_list), self.num_topics))))
            #from sklearn.preprocessing import normalize
            #normalize(doc_prof, norm='l1', axis=1, copy=False)
            return doc_prof


class TWCNB(BaseEstimator, TransformerMixin):

    """ Models that extracts Second Order Attributes
     based on Rennie, Shih, Teevan and Karger </Paper>"""

    def __init__(self, max_df=1.0, min_df=5, tokenizer_var = 'sklearn', max_features=None):
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        

        # stop_list = []
        # with open(stopwords_path, 'r') as stop_inp:
        # for w in stop_inp:
        # stop_list.append(w.replace("\n", ""))
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.tokenizer_var = tokenizer_var
        self.term_table = None
        self.labels = None
        if self.tokenizer_var == '1':
            self.tokenization = tokenization
        elif self.tokenizer_var == '2':
            self.tokenization = tokenization2
        elif self.tokenizer_var == '3':
            self.tokenization = _twokenize.tokenizeRawTweetText
        else:
            self.tokenization = None
        # self.lsi = None
        # self.dictionary = None
        # self.num_topics = 100
        # self.counter = CountVectorizer()
        self.counter = TfidfVectorizer(sublinear_tf=True)

    def fit(self, X, y=None):

        import numpy
        from sklearn.preprocessing import normalize

        # print "We are fitting!"
        if y is None:
            raise ValueError('we need y labels to supervise-fit!')
        else:
            # texts = [self.tokenization(text) for text in X]
            # self.dictionary = corpora.Dictionary(texts)
            # corpus = [self.dictionary.doc2bow(text) for text in texts]
            # self.lsi = models.LsiModel(corpus, id2word=self.dictionary, num_topics=self.num_topics)
            # for token in tokens:
            #    voc = voc.union(token)
            # print len(voc)
            # print list(voc)[:100]
            parameters = {
                'input': 'content',
                'encoding': 'utf-8',
                'decode_error': 'ignore',
                'analyzer': 'word',
                # 'vocabulary':list(voc),
                'tokenizer': self.tokenization,
                #'tokenizer': _twokenize.tokenizeRawTweetText,  # self.tokenization,
                #'tokenizer': lambda text: _twokenize.tokenizeRawTweetText(nonan.sub(po_re.sub('', text))),
                'max_df': self.max_df,
                'min_df': self.min_df,
                'max_features': self.max_features
            }
            self.counter.set_params(**parameters)
            # print len(target_profiles)
            doc_term = self.counter.fit_transform(X)
            # print "New one2"
            #normalize(doc_term, norm='l2', axis=1, copy=False)
            # print "Doc_Terms"
            # print doc_term.shape, type(doc_term)
            target_profiles = sorted(list(set(y)))
            self.labels = target_profiles
            doc_prof = numpy.zeros([doc_term.shape[0], len(target_profiles)])
            for i in range(0, doc_term.shape[0]):
                # tmp = numpy.zeros([1, len(target_profiles)])
                tmp = numpy.ones([1, len(target_profiles)])
                tmp[0, target_profiles.index(y[i])] = 0
                doc_prof[i, :] = tmp
            # print "Doc_Prof"
            # print doc_prof.shape, type(doc_prof)
            #doc_term.data = numpy.log2(doc_term.data + 1)
            #doc_term.transpose
            # print "Doc_Term"
            # print doc_term.shape, type(doc_term)
            nominator = doc_term.transpose().dot(doc_prof)
            # LAPLACE SMOOTHING
            a = 1
            nominator = nominator + a
            # print "Term_Prof"
            # print nominator.shape, type(nominator)
            doc_sum = doc_term.sum(axis=1)
            doc_sum = numpy.array(doc_sum, copy=False)
            # print "Doc_Sum"
            # print doc_sum.shape, type(doc_sum)
            basic_row = numpy.dot(doc_sum.T, doc_prof)
            basic_row = basic_row + a*doc_term.shape[1]
            # print "Basic_Row"
            # print basic_row.shape, type(basic_row)
            denominator = numpy.tile(basic_row, (nominator.shape[0],1))
            # print "Denominator"
            # print denominator.shape, type(denominator)

            #term_prof = numpy.zeros([doc_term.shape[1], len(target_profiles)])
            #term_prof = numpy.log2(doc_term.transpose.data
            #term_prof = numpy.dot(
            #    numpy.log2(doc_term.toarray().astype('float', casting='unsafe').T + 1), doc_prof)
            
            # normalize against words
            # term_prof = term_prof / term_prof.sum(axis=0)
            # normalize across profiles
            # term_prof = term_prof / \
                # numpy.reshape(
                   # term_prof.sum(axis=1), (term_prof.sum(axis=1).shape[0], 1))
            # term_prof = term_prof / \
            #    numpy.reshape(
            #        term_prof.sum(axis=1), (term_prof.sum(axis=1).shape[0], 1))
            # term_prof = term_prof / term_prof.sum(axis=0)
            self.term_table = numpy.log2(nominator*denominator)# term_prof
            self.term_table = normalize(self.term_table, norm='l1', axis=1, copy=False)
            print "Random Term_Prof"
            # print self.counter.vocabulary_
            print self.term_table[0,:]
            # print "SOA Model Fitted!"
            return self

    def transform(self, X, y=None):

        import numpy
        from sklearn.preprocessing import normalize

        # print "We are transforming!"
        if self.labels is None:
            raise AttributeError('term_table was no found! \
             Probably model was not fitted first. Run model.fit(X,y)!')
        else:
            #doc_term = numpy.zeros(
            #    [len(X), self.term_table.shape[0]])
            doc_term = self.counter.transform(X)
            normalize(doc_term, norm='l2', axis=1, copy=False)
            # print "Doc_Terms"
            # print doc_term.shape, type(doc_term)
            doc_prof = numpy.zeros(
                [doc_term.shape[0], self.term_table.shape[1]])
            # print "Term_Prof"
            # print self.term_table.shape
            doc_prof = doc_term.dot(self.term_table)
            # doc_prof = numpy.dot(
            #    doc_term.toarray().astype('float', casting='unsafe'),
            #    self.term_table)
            # print "SOA Transform:"
            # print type(doc_prof)
            # print 'Doc_prof'
            # print doc_prof.shape, type(doc_prof)
            #print "Len Voc: %s\n" % (str(doc_term.shape[1]))
            #import pprint
            #pprint.pprint(self.counter.vocabulary_)
            # LSI
            # texts = [self.tokenization(text) for text in X]
            # corpus = [self.dictionary.doc2bow(text) for text in texts]
            # transform_lsi = self.lsi[corpus]
            # lsi_list = []
            # dummy_empty_list = [0 for i in range(0, self.num_topics)]
            # #c = 0
            # for i, doc in enumerate(transform_lsi):
            #     if not doc:  # list is empty
            #         lsi_list.append(dummy_empty_list)
            #     else:
            #         lsi_list.append(list(zip(*doc)[1]))
            #         if len(lsi_list[-1]) != self.num_topics:
            #             # c += 1
            #             # print c
            #             # print texts[i]
            #             # print len(lsi_list[-1])
            #             # print lsi_list[-1]
            #            lsi_list[-1] = dummy_empty_list
            # lsi_list = [list(zip(*doc)[1]) for doc in transform_lsi]
            # print numpy.array(lsi_list).shape
            # print len(lsi_list)
            # print len(lsi_list[0])
            # c = numpy.reshape(numpy.array(lsi_list), (len(lsi_list), self.num_topics))
            # print c.shape
            # return numpy.hstack((doc_prof, numpy.reshape(numpy.array(lsi_list), (len(lsi_list), self.num_topics))))
            return doc_prof




class LSI_Model(BaseEstimator, TransformerMixin):
    """ Model that extracts LSI features"""

    def __init__(self, num_topics=100):

        # stop_list = []
        # with open(stopwords_path, 'r') as stop_inp:
        # for w in stop_inp:
        # stop_list.append(w.replace("\n", ""))
        self.lsi = None
        self.dictionary = None
        self.num_topics = num_topics

    def fit(self, X, y=None):

        from gensim import corpora, models

        #print "We are fitting!"
        if y is None:
            raise ValueError('we need y labels to supervise-fit!')
        else:
            texts = [tokenization(text) for text in X]
            self.dictionary = corpora.Dictionary(texts)
            corpus = [self.dictionary.doc2bow(text) for text in texts]
            self.lsi = models.LsiModel(corpus, id2word=self.dictionary, num_topics=self.num_topics)
            #print "LSI Model Fitted!"
            #print "Dict len: %s" % (len(self.dictionary.values()))
            # import pprint
            # print "Dict:"
            # pprint.pprint(sorted(self.dictionary.values()))
            return self

    def transform(self, X, y=None):

        import numpy

        #print "We are transforming!"
        if self.lsi is None:
            raise AttributeError('lsi_model was no found! \
             Probably model was not fitted first. Run model.fit(X,y)!')
        else:
            # LSI
            texts = [tokenization(text) for text in X]
            corpus = [self.dictionary.doc2bow(text) for text in texts]
            transform_lsi = self.lsi[corpus]
            lsi_list = []
            dummy_empty_list = [0 for i in range(0, self.num_topics)]
            # c = 0
            for i, doc in enumerate(transform_lsi):
                if not doc:  # list is empty
                    lsi_list.append(dummy_empty_list)
                else:
                    lsi_list.append(list(zip(*doc)[1]))
                    if len(lsi_list[-1]) != self.num_topics:
                        # c += 1
                        # print c
                        # print texts[i]
                        # print len(lsi_list[-1])
                        # print lsi_list[-1]
                        lsi_list[-1] = dummy_empty_list
            # lsi_list = [list(zip(*doc)[1]) for doc in transform_lsi]
            # print numpy.array(lsi_list).shape
            # print len(lsi_list)
            temp_z = numpy.reshape(numpy.array(lsi_list), (len(lsi_list), self.num_topics))
            #print "LSI Transform:"
            #print temp_z.shape
            # print len(lsi_list[0])
            # for Naive Bayes to have only semi-positive values
            return temp_z + abs(temp_z.min())

""" 
    def predict(self, X, y=None):

        import numpy
        print "We are predicting!"
        doc_prof = self.transform(X)
        y_pred = []
        for i in range(0, doc_prof.shape[0]):
            y_pred.append(self.labels[numpy.argmax(doc_prof[i, :])])
        return y_pred """

class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, K=100):
        """ Initialize max class document
        """
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2
        
        self.K = K
        self.Kbest = SelectKBest(score_func=chi2, K=self.K)

    def fit(self, X, y):
        X_new = self.Kbest.fit_transform(X, y)
        return self

    def transform(self, X, y=None):
        return self.Kbest.transform(X)



class SOA_Predict(object):

    def __init__(self):
        """ Initialize max class document
        """
        self.help = self.__doc__
        self.labels = None

    def fit(self, X, y, sample_weight=None):
        target_profiles = sorted(list(set(y)))
        self.labels = target_profiles
        return self

    def score(self, X, y_true):

        from sklearn.metrics import accuracy_score

        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred, normalize=True)

    def predict(self, doc_prof):
        import pprint
        import numpy

        y_pred = []
        # print type(doc_prof)
        #pprint.pprint(doc_prof)
        for i in range(0, doc_prof.shape[0]):
            # y_pred.append(self.labels[numpy.argmax(doc_prof[i, :])])
            y_pred.append(self.labels[numpy.argmin(doc_prof[i, :])])
        return y_pred


class LDA(BaseEstimator, TransformerMixin):

    """ LDA MODELS """

    def __init__(self, num_topics=100, lib='sklearn'):

        from sklearn.feature_extraction.text import CountVectorizer

        self.num_topics = num_topics
        self.lib = lib
        print self.lib
        self.labels = None
        self.corpus = None
        self.dictionary = None
        self.counter = CountVectorizer()
        if self.lib == 'sklearn':
            from sklearn.decomposition import LatentDirichletAllocation
            self.LDA = LatentDirichletAllocation(n_topics=self.num_topics)
        else:
            self.LDA = None

    def fit(self, X, y=None):

        if y is None:
            raise ValueError('we need y labels to supervise-fit!')
        else:
            target_profiles = sorted(list(set(y)))
            self.labels = target_profiles
            if self.LDA is None:
                from gensim import corpora, models
                X = [text.lower().split() for text in X]
                self.dictionary = corpora.Dictionary(X)
                self.corpus = [self.dictionary.doc2bow(text) for text in X]
                if self.lib == 'gensim':
                    self.LDA = models.LdaModel(num_topics=self.num_topics, corpus=self.corpus, id2word=self.dictionary, minimum_probability=0.00)
                elif self.lib == 'mallet':
                    self.LDA = models.wrappers.LdaMallet('/home/kostas/Downloads/mallet-2.0.7/bin/mallet', corpus=self.corpus, num_topics=self.num_topics, id2word=self.dictionary)
            else:
                parameters = {
                    'input': 'content',
                    'encoding': 'utf-8',
                    'decode_error': 'ignore',
                    'analyzer': 'word',
                    'stop_words': 'english',
                    # 'vocabulary':list(voc),
                    #'tokenizer': tokenization,
                    #'tokenizer': _twokenize.tokenizeRawTweetText,  # self.tokenization,
                    #'tokenizer': lambda text: _twokenize.tokenizeRawTweetText(nonan.sub(po_re.sub('', text))),
                    'max_df': 0.8,
                    'min_df': 5,
                    'max_features': 5000
                }
                self.counter.set_params(**parameters)
                doc_term = self.counter.fit_transform(X)
                self.LDA.fit(doc_term, y)
            return self

    def transform(self, X, y=None):

        # print "We are transforming!"
        if self.labels is None:
            raise AttributeError('term_table was no found! \
             Probably model was not fitted first. Run model.fit(X,y)!')
        else:
            #doc_term = numpy.zeros(
            #    [len(X), self.term_table.shape[0]])
            if self.lib == 'sklearn':
                doc_term = self.counter.transform(X)
                doc_topics = self.LDA.transform(doc_term)
                #print("\nTopics in LDA model:")
                tf_feature_names = self.counter.get_feature_names()
                #print_top_words(self.LDA, tf_feature_names, 10)
            else:
                X = [text.lower().split() for text in X]
                test_corpus = [self.dictionary.doc2bow(text) for text in X]
                if self.LDA is None:
                    print self.lib
                    print 'dic'
                    #print self.__dict__
                doc_topics1 = self.LDA[test_corpus]
                doc_topics1 = [[topic[1] for topic in doc] for doc in doc_topics1]
                doc_topics = numpy.array(doc_topics1)
                #print("\nTopics in LDA model:")
                #self.LDA.print_topics(self.num_topics, 10)
            return doc_topics







class skLDA(BaseEstimator, TransformerMixin):

    """ LDA model based on sklearnLDA"""

    def __init__(self, n_topics=100, verbose=1, random_state=42):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        self.n_topics = n_topics
        self.verbose = verbose
        self.random_state = random_state
        # print "num topics:" + str(n_topics)
        # print "verbose:" + str(verbose)
        self.labels = None
        # bazw manually ta numtopics ktlp giati pernane san None Orismata..Vale ta print an thes..
        self.LDA = LatentDirichletAllocation(n_topics=self.n_topics, verbose=self.verbose, random_state=self.random_state)
        # Conceptually much better results with TFIDFVECTORIZER(use_log tf)
        self.counter = CountVectorizer()

    def fit(self, X, y=None):

        # print "We are fitting!"
        if y is None:
            raise ValueError('we need y labels to supervise-fit!')
        else:
            parameters = {
                'input': 'content',
                'encoding': 'utf-8',
                'decode_error': 'ignore',
                'analyzer': 'word',
                # 'vocabulary':list(voc),
                #'tokenizer': tokenization,
                #'tokenizer': _twokenize.tokenizeRawTweetText,  # self.tokenization,
                #'tokenizer': lambda text: _twokenize.tokenizeRawTweetText(nonan.sub(po_re.sub('', text))),
                'max_df': 1.0,
                'min_df': 5,
                'max_features': None
            }
            self.counter.set_params(**parameters)
            doc_term = self.counter.fit_transform(X)
            target_profiles = sorted(list(set(y)))
            self.labels = target_profiles
            self.LDA.fit(doc_term, y)
            return self

    def transform(self, X, y=None):

        # print "We are transforming!"
        if self.labels is None:
            raise AttributeError('term_table was no found! \
             Probably model was not fitted first. Run model.fit(X,y)!')
        else:
            #doc_term = numpy.zeros(
            #    [len(X), self.term_table.shape[0]])
            doc_term = self.counter.transform(X)
            doc_topics = self.LDA.transform(doc_term)
            print("\nTopics in LDA model:")
            tf_feature_names = self.counter.get_feature_names()
            print_top_words(self.LDA, tf_feature_names, 20)
            return doc_topics


def print_top_words(model, feature_names, n_top_words):

    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


class skNMF(BaseEstimator, TransformerMixin):

    """ NMF model based on sklearn"""

    def __init__(self, n_components=100, verbose=1, random_state=42):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import NMF
        
        self.n_components = n_components
        self.verbose = verbose
        self.random_state = random_state
        # print "verbose:" + str(verbose)
        self.labels = None
        self.NMF = NMF(init='nndsvd', n_components=self.n_components, verbose=self.verbose, random_state=self.random_state)
        # Conceptually much better results with TFIDFVECTORIZER(use_log tf)
        self.counter = TfidfVectorizer(sublinear_tf=True)

    def fit(self, X, y=None):


        # print "We are fitting!"
        if y is None:
            raise ValueError('we need y labels to supervise-fit!')
        else:
            parameters = {
                'input': 'content',
                'encoding': 'utf-8',
                'decode_error': 'ignore',
                'analyzer': 'word',
                # 'vocabulary':list(voc),
                #'tokenizer': tokenization,
                #'tokenizer': _twokenize.tokenizeRawTweetText,  # self.tokenization,
                #'tokenizer': lambda text: _twokenize.tokenizeRawTweetText(nonan.sub(po_re.sub('', text))),
                'max_df': 1.0,
                'min_df': 5,
                'max_features': None
            }
            self.counter.set_params(**parameters)
            doc_term = self.counter.fit_transform(X)
            target_profiles = sorted(list(set(y)))
            self.labels = target_profiles
            self.NMF.fit(doc_term, y)
            return self

    def transform(self, X, y=None):

        # print "We are transforming!"
        if self.labels is None:
            raise AttributeError('term_table was no found! \
             Probably model was not fitted first. Run model.fit(X,y)!')
        else:
            #doc_term = numpy.zeros(
            #    [len(X), self.term_table.shape[0]])
            doc_term = self.counter.transform(X)
            doc_topics = self.NMF.transform(doc_term)
            tf_feature_names = self.counter.get_feature_names()
            print_top_words(self.NFM, tf_feature_names, 20)
            return doc_topics


class XGBoostClassifier(BaseEstimator, TransformerMixin):

    def __init__(self, **params):

        self.clf = None
        # self.num_boost_round = 100
        self.labels = None
        self.params = params.copy()
        for name, label in params.iteritems():
            setattr(self, name, label)

    def fit(self, X, y, num_boost_round=None):

        import xgboost as xgb

        num_boost_round = num_boost_round or self.num_boost_round
        print self.params
        print self.__dict__
        print num_boost_round
        self.label2num = dict((label, i) for i, label in enumerate(sorted(set(y))))
        self.params.update({'num_class': len(set(y))})
        print type(self.params)
        print self.params
        # self.params.update({'objective': 'multi:softprob', 
        #                     'num_class': len(set(y)), 
        #                     'eval_metric': 'merror', 
        #                     'nthread': -1, 
        #                     'learning_rate': 0.1, 
        #                     'n_estimators': 140, 
        #                     'max_depth': 5,
        #                     'min_child_weight': 1, 
        #                     'gamma': 0, 
        #                     'subsample': 0.8, 
        #                     'colsample_bytree': 0.8,
        #                     'scale_pos_weight': 1, 
        #                     'seed': 27})
        if y is None:
            raise ValueError('we need y labels to supervise-fit!')
        else:
            dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
            self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round)

    def predict(self, X):

        import numpy

        num2label = dict((i, label)for label, i in self.label2num.items())
        Y = self.predict_proba(X)
        y = numpy.argmax(Y, axis=1)
        return numpy.array([num2label[i] for i in y])

    def predict_proba(self, X):

        from xgboost import DMatrix

        dtest = DMatrix(X)
        return self.clf.predict(dtest)

    def score(self, X, y_true):

        from sklearn.metrics import accuracy_score

        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred, normalize=True)

        # Y = self.predict_proba(X)
        # return 1 / logloss(y, Y)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self


class Metaclassifier(BaseEstimator, TransformerMixin):

    """ A Linear Weights Metaclassifier """

    def __init__(self, models, C=1.0, weights='balanced'):

        from sklearn.svm import LinearSVC
        from sklearn.preprocessing import LabelEncoder

        if not models:
            raise AttributeError('Models expexts a dictonary of models \
              containg the predictions of y_true for each classifier')
        self.models = models
        self.weights = weights
        self.C = C
        self.svc = LinearSVC(C=self.C, class_weight=self.weights)
        self.lab_encoder = LabelEncoder()

    def fit(self, X_cv, y_true=None, weights=None):

        if y_true is None:
            raise ValueError('we need y labels to supervise-fit!')
        else:
            # import pprint
            #print list(set(y_true))
            # print len(y_true)
            y_true = self.lab_encoder.fit_transform(y_true)
            #print self.models.keys()
            # print self.lab_encoder.classes_
            # print self.models[self.models.keys()[1]].predict(X_cv)
            #y_true = self.create_onehot(y_true)
            # print "Train X shape: " + str(X_cv.shape) + "train y_true " + str(y_true.shape)
            transformed_y = self.transform_to_y(X_cv)
            #X = self.oh_encoder.transform(y_pred.T)
            # print transformed_y.shape, y_true.T.shape
            #print "fit true"
            # print transformed_y
            # print y_true
            self.svc.fit(transformed_y, y_true.T)
            return self

    def predict(self, X):

        # print "PRedict"
        # print X.shape
        X = self.transform_to_y(X)
        # print "PRedict after"
        # print X.shape
        # print X.T.shape
        import pprint
        # pprint.pprint(X)
        # pprint.pprint(X.T)
        #print "Predict"
        y_pred = self.svc.predict(X)
        #pprint.pprint(y_pred)
        #pprint.pprint(self.lab_encoder.inverse_transform(y_pred))
        return self.lab_encoder.inverse_transform(y_pred)

    def score(self, X, y, sample_weight=None):

        # import numpy
        # print "Score"
        # print X.shape, numpy.array(y).shape
        # transformed_y = self.transform_to_y(X)
        # print 'edw ok'
        # print self.svc.predict(transformed_y).shape
        # print 'Transformed'
        # print transformed_y.shape
        from sklearn.metrics import accuracy_score
        import pprint
        #print "Ture"
        #pprint.pprint(y)

        return accuracy_score(y, self.predict(X), normalize=True)
        #return self.svc.score(self.transform_to_y(X), y, sample_weight)

    def create_onehot(self, l):

        from numpy import zeros, vstack
        #print "L:"
        #from pprint import pprint as pprint
        #print type(l)
        # pprint(l)
        l = list(l)
        for i, el in enumerate(l):
            temp = zeros([1, len(self.lab_encoder.classes_)], dtype=float)
            #print(temp.shape)
           # pprint(temp)
            temp[0, el] = 1
            if i == 0:
                fin = temp
            else:
                fin = vstack((fin, temp))
        #print "onehot shape" + str(fin.shape)
        return fin

    def transform_to_y(self, X):

        from numpy import hstack

        # print "Train X shape: " + str(X.shape)
        for i, model in enumerate(self.models.values()):
            # print self.models.keys()[i]
            tmp_pred = self.create_onehot(self.lab_encoder.transform(model.predict(X)))
            if i == 0:
                y_pred = tmp_pred
            else:
                y_pred = hstack((y_pred, tmp_pred))
        # print "y_pred: " + str(y_pred.shape)
        return y_pred


class Metaclassifier2(BaseEstimator, TransformerMixin):

    """ A Linear Weights Metaclassifier based on the neighborhood of each sample.
        The neighborhood is different per base model. For each sample we have
        [N, N*k] votes, with N the number of base classifiers and k the number
        of neighbors to look for. """

    def __init__(self, models, C=1.0, weights='balanced', k=3):

        from sklearn.svm import LinearSVC
        from sklearn.preprocessing import LabelEncoder

        if not models:
            raise AttributeError('Models expexts a dictonary of models \
              containg the predictions of y_true for each classifier')
        self.models = models
        self.weights = weights
        self.C = C
        self.k = 3
        self.svc = LinearSVC(C=self.C, class_weight=self.weights)
        self.lab_encoder = LabelEncoder()

    def fit(self, X_cv, y_true=None, weights=None):

        if y_true is None:
            raise ValueError('we need y labels to supervise-fit!')
        else:
            # import pprint
            #print list(set(y_true))
            # print len(y_true)
            y_true = self.lab_encoder.fit_transform(y_true)
            #print self.models.keys()
            print self.lab_encoder.classes_
            # print self.models[self.models.keys()[1]].predict(X_cv)
            #y_true = self.create_onehot(y_true)
            # print "Train X shape: " + str(X_cv.shape) + "train y_true " + str(y_true.shape)
            transformed_y = self.transform_to_y(X_cv)
            #X = self.oh_encoder.transform(y_pred.T)
            #print transformed_y.shape, y_true.T.shape
            #print "fit true"
            # print transformed_y
            # print y_true
            self.svc.fit(transformed_y, y_true.T)
            return self

    def predict(self, X):

        # print "PRedict"
        # print X.shape
        X = self.transform_to_y(X)
        # print "PRedict after"
        # print X.shape
        # print X.T.shape
        import pprint
        # pprint.pprint(X)
        # pprint.pprint(X.T)
        # print "Predict"
        y_pred = self.svc.predict(X)
        # pprint.pprint(y_pred)
        # pprint.pprint(self.lab_encoder.inverse_transform(y_pred)) 
        return self.lab_encoder.inverse_transform(y_pred)

    def score(self, X, y, sample_weight=None):

        # import numpy
        # print "Score"
        # print X.shape, numpy.array(y).shape
        # transformed_y = self.transform_to_y(X)
        # print 'edw ok'
        # print self.svc.predict(transformed_y).shape
        # print 'Transformed'
        # print transformed_y.shape
        from sklearn.metrics import accuracy_score
        # import pprint
        # print "Ture"
        # pprint.pprint(y)

        return accuracy_score(y, self.predict(X), normalize=True)
        #return self.svc.score(self.transform_to_y(X), y, sample_weight)

    def create_onehot(self, l):

        from numpy import zeros, vstack
        #print "L:"
        #from pprint import pprint as pprint
        #print type(l)
        # pprint(l)
        l = list(l)
        for i, el in enumerate(l):
            temp = zeros([1, len(self.lab_encoder.classes_)], dtype=float)
            #print(temp.shape)
           # pprint(temp)
            temp[0, el] = 1
            if i == 0:
                fin = temp
            else:
                fin = vstack((fin, temp))
        #print "onehot shape" + str(fin.shape)
        return fin

    def transform_to_y(self, X):

        from numpy import hstack

        #print "Train X shape: " + str(X.shape)
        for i, model in enumerate(self.models.values()):
            #print self.models.keys()[i]
            predict = model.predict(X)
            #print type(predict)
            #print predict.shape
            #print predict
            tmp_pred = self.create_onehot(self.lab_encoder.transform(predict))
            #print type(tmp_pred)
            if i == 0:
                y_pred = tmp_pred
            else:
                y_pred = hstack((y_pred, tmp_pred))
            predictions_n = self.neigh_model_pred(model, X, predict)
            #print 'Num Pred'
            #print len(predictions_n[0])
            for neigh_dist in xrange(self.k):
                tmp_pred_n = self.create_onehot(self.lab_encoder.transform(predictions_n[:, neigh_dist]))
                y_pred = hstack((y_pred, tmp_pred_n))
            #print "y_pred: " + str(y_pred.shape)
        #print "y_pred: " + str(y_pred.shape)
        #print len(self.lab_encoder.classes_)
        #print y_pred
        return y_pred


    def neigh_model_pred(self, modle, X, pred):

        from sklearn.neighbors import BallTree
        import numpy

        # Expects a pipeline with two steps. Transform and Predict.
        transf = model.steps[0][1].transform(X)
        if hasattr(transf, "toarray"):
            # print 'Exei'
            representations = transf.toarray()
        else:
            representations = transf
        ModelTree = BallTree(representations)
        predictions = []
        for i in xrange(representations.shape[0]):
            _, neig_ind = ModelTree.query(representations[i,:].reshape(1,-1), self.k)
            predictions.extend([pred[n_i] for n_i in neig_ind])
        return numpy.array(predictions)
