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
        import lda
        from sklearn.feature_extraction.text import CountVectorizer
        self.n_topics = n_topics
        self.k_top = k_top
        self.model = lda.LDA(n_topics=self.n_topics,
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
        return [[len(CountHash.pat.findall(text))] for text in texts]


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
        return [[len(CountReplies.pat.findall(text))] for text in texts]


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
        return [[len(CountURLs.pat.findall(text))] for text in texts]


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

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        """ transform data

        :texts: The texts to count capital words in
        :returns: list of counts for each text

        """
        l = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
             'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
             'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3',
             '4', '5', '6', '7', '8', '9', '!', '.', ':', '?',
             ';', ',', ')', '(', '-', '%', '$', '#', '@', '^',
             '&', '*', '=', '+', '/', '"', "'", '<', '>', '|',
             '~', '`']
        return [[text.lower().count(token) for token in l]
                for text in texts]

# class SOA_Model2(object):


class SOA_Model2(BaseEstimator, TransformerMixin):

    """ Models that extracts Second Order Attributes
     (SOA) base on PAN 2013-2015 Winners"""

    def __init__(self, max_df=1.0, min_df=5, max_features=None):
        from sklearn.feature_extraction.text import TfidfVectorizer

        # stop_list = []
        # with open(stopwords_path, 'r') as stop_inp:
        # for w in stop_inp:
        # stop_list.append(w.replace("\n", ""))
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.term_table = None
        self.labels = None
        # self.lsi = None
        # self.dictionary = None
        # self.num_topics = 100
        # self.counter = CountVectorizer()
        self.counter = TfidfVectorizer(use_idf=False)

    def fit(self, X, y=None):

        import numpy
        from sklearn.preprocessing import StandardScaler, normalize

        print "We are fitting!"
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
                # 'tokenizer': tokenization,
                #'tokenizer': _twokenize.tokenizeRawTweetText,  # self.tokenization,
                #'tokenizer': lambda text: _twokenize.tokenizeRawTweetText(nonan.sub(po_re.sub('', text))),
                'max_df': self.max_df,
                'min_df': self.min_df,
                'max_features': self.max_features
            }
            self.counter.set_params(**parameters)
            print str(self.counter.get_params())
            # print len(target_profiles)
            doc_term = self.counter.fit_transform(X)
            # st_scaler = StandardScaler(copy=False)
            # st_scaler.fit_transform(doc_term)
            #normalize(doc_term, norm='l1', axis=0, copy=False)
            print "Doc_Terms"
            print doc_term.shape
            target_profiles = sorted(list(set(y)))
            self.labels = target_profiles
            doc_prof = numpy.zeros([doc_term.shape[0], len(target_profiles)])
            for i in range(0, doc_term.shape[0]):
                tmp = numpy.zeros([1, len(target_profiles)])
                tmp[0, target_profiles.index(y[i])] = 1
                doc_prof[i, :] = tmp
            print "Doc_Prof"
            print doc_prof.shape, type(doc_prof)
            doc_term.data = numpy.log2(doc_term.data + 1)
            #doc_term.transpose
            print "Doc_Term"
            print doc_term.shape, type(doc_term)
            term_prof = doc_term.transpose().dot(doc_prof)
            #term_prof = numpy.zeros([doc_term.shape[1], len(target_profiles)])
            #term_prof = numpy.log2(doc_term.transpose.data
            #term_prof = numpy.dot(
            #    numpy.log2(doc_term.toarray().astype('float', casting='unsafe').T + 1), doc_prof)
            print "Term_Prof"
            print term_prof.shape, type(term_prof)
            # normalize against words
            term_prof = term_prof / term_prof.sum(axis=0)
            # normalize across profiles
            term_prof = term_prof / \
                numpy.reshape(
                   term_prof.sum(axis=1), (term_prof.sum(axis=1).shape[0], 1))
            print "Random Term_Prof"
            print term_prof[0,:]
            # term_prof = term_prof / \
            #    numpy.reshape(
            #        term_prof.sum(axis=1), (term_prof.sum(axis=1).shape[0], 1))
            # term_prof = term_prof / term_prof.sum(axis=0)
            self.term_table = term_prof
            print "SOA Model Fitted!"
            return self

    def transform(self, X, y=None):

        import numpy

        print "We are transforming!"
        if self.labels is None:
            raise AttributeError('term_table was no found! \
             Probably model was not fitted first. Run model.fit(X,y)!')
        else:
            #doc_term = numpy.zeros(
            #    [len(X), self.term_table.shape[0]])
            doc_term = self.counter.transform(X)
            print "Doc_Terms"
            print doc_term.shape, type(doc_term)
            doc_prof = numpy.zeros(
                [doc_term.shape[0], self.term_table.shape[1]])
            # print "Term_Prof"
            # print self.term_table.shape
            doc_prof = doc_term.dot(self.term_table)
            # doc_prof = numpy.dot(
            #    doc_term.toarray().astype('float', casting='unsafe'),
            #    self.term_table)
            print "SOA Transform:"
            # print type(doc_prof)
            print 'Doc_prof'
            print doc_prof.shape, type(doc_prof)
            print "Len Voc: %s" % (str(len(self.counter.vocabulary_)))
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
            return doc_prof


class TWCNB(BaseEstimator, TransformerMixin):

    """ Models that extracts Second Order Attributes
     based on Rennie, Shih, Teevan and Karger </Paper>"""

    def __init__(self, max_df=1.0, min_df=5, max_features=None):
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        

        # stop_list = []
        # with open(stopwords_path, 'r') as stop_inp:
        # for w in stop_inp:
        # stop_list.append(w.replace("\n", ""))
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.term_table = None
        self.labels = None
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
                #'tokenizer': tokenization,
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
            print "Len Voc: %s\n" % (str(doc_term.shape[1]))
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

        print "We are fitting!"
        if y is None:
            raise ValueError('we need y labels to supervise-fit!')
        else:
            texts = [tokenization(text) for text in X]
            self.dictionary = corpora.Dictionary(texts)
            corpus = [self.dictionary.doc2bow(text) for text in texts]
            self.lsi = models.LsiModel(corpus, id2word=self.dictionary, num_topics=self.num_topics)
            print "LSI Model Fitted!"
            print "Dict len: %s" % (len(self.dictionary.values()))
            # import pprint
            # print "Dict:"
            # pprint.pprint(sorted(self.dictionary.values()))
            return self

    def transform(self, X, y=None):

        import numpy

        print "We are transforming!"
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
            print "LSI Transform:"
            print temp_z.shape
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
            #y_pred.append(self.labels[numpy.argmax(doc_prof[i, :])])
            y_pred.append(self.labels[numpy.argmin(doc_prof[i, :])])
            if i == 0:
                print y_pred
                print doc_prof[i, :]
        return y_pred
