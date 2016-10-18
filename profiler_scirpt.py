import sys
sys.path.insert(0, "../EnsembleDiversityTests/")
import combinations
import copy
import time
import numpy
import warnings
from pan import ProfilingDataset
from pan import preprocess
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from pan.features import LSI_Model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,precision_recall_fscore_support, classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.grid_search import GridSearchCV
from pan.features import Metaclassifier
from pan import features
from functools import wraps
from memory_profiler import profile

warnings.filterwarnings('ignore')


# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix





class SubSpaceEnsemble3(BaseEstimator, TransformerMixin):
    
    """ Utilizing the neighborhood in all representations and also ground truth model.
        Implementing a weighted voting scheme."""

    def __init__(self, models, k=3, weights= [2,1,3,0.7]):
        from sklearn.feature_extraction.text import CountVectorizer
        
        if (not models):
            raise AttributeError('Models expexts a dictonary of models \
              containg the predictions of y_true for each classifier.\ ')
        else:
            self.models = models
            # self.cv_scores = cv_scores
            self.k = k
            self.weights = weights
            self.ind2names = {}
            for i, name in enumerate(models.keys()):
                self.ind2names[i] = name
            self.counter = CountVectorizer()
            self.representations = []
            self.meta = None
            self.predictions = []
            self.true = []
            self.doc_terms = None
            self.tree = None
            self.experts = []
        

    def fit(self, X_cv, y_true=None, weights=None):
        
        from sklearn.neighbors import BallTree
        import random

        if y_true is None:
            raise ValueError('we need y labels to supervise-fit!')
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
                    'max_df': 1.0,
                    'min_df': 1,
                    'max_features':None
                }
            t0 = time.time()
            self.counter.set_params(**parameters)
            self.doc_terms = self.counter.fit_transform(X_cv).toarray()
            self.tree = BallTree(self.doc_terms, leaf_size=20)
            predictions = []
            for name, model in self.models.iteritems():
                predictions.append(model.predict(X_cv))
                #print len(predictions[-1])
                transf = model.steps[0][1].transform(X_cv)
                if hasattr(transf, "toarray"):
                    #print 'Exei'
                    self.representations.append(transf.toarray())
                else:
                    self.representations.append(transf)
            self.predictions = predictions
            self.true = y_true
            count = 0
            #print self.expert_scores
            #print self.experts
            print('Fit took: %0.3f seconds') % (time.time()-t0)
            return self

    def predict(self, X):
        # print "PRedict"
        # print X.shape
        X_transformed = self.counter.transform(X).toarray()
        #print type((X_transformed)[0])
        #print X_transformed.shape
        #return 0
        y_pred = []
        t0 = time.time()
        for i in range(0, X_transformed.shape[0]):
            #print X_transformed[i,:].shape
            dist, neigbors_indexes = self.tree.query(X_transformed[i,:].reshape(1,-1), self.k)  
            #print 'Sample ' + y_real[i]
            #print dist
            #print type(dist)
            #print neigbors_indexes[0]
            #print dist
            #best_model_ind = self.expert_decision(neigbors_indexes[0])
            #pass
            y_pred.append(self.expert_decision(neigbors_indexes[0],  dist, X[i]))
            
            #y_pred.append(self.models[self.ind2names[best_model_ind]].predict([X[i]])[0])
        #print y_pred
        print('Predict took: %0.3f seconds') % (time.time()-t0)
        return y_pred

    def score(self, X, y, sample_weight=None):

        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), normalize=True)
        #return self.svc.score(self.transform_to_y(X), y, sample_weight)


    def expert_decision(self, neigbors_indexes, dist, x_sample):

        from sklearn.metrics import accuracy_score
        from collections import Counter
        from sklearn.neighbors import BallTree
        
        models_pred = []
        models_neig_pred = []
        acc = []
        t0 = time.time()
        neigbors_true = [self.true[n_i] for n_i in neigbors_indexes]
        #print('Neighbors per sample: %0.4f seconds') % (time.time()-t0)
#         print 'True'
#         print neigbors_true
        sample_predictions = []
        total_pred = []
        weights = {}
        weights['true'] = self.weights[2]
        weights['models_n'] = []
        weights['models'] = []
        for model_i in xrange(len(self.models.values())):
            ModelTree = BallTree(self.representations[model_i])
            temp_trans = self.models[self.ind2names[model_i]].steps[0][1].transform([x_sample])
            if hasattr(temp_trans, 'toarray'):
                temp_trans = temp_trans.toarray()
            _, model_neig = ModelTree.query(temp_trans, self.k)
            model_neig_pred = []
            for model_n_i in model_neig[0].tolist():
                model_neig_pred.append(self.predictions[model_i][model_n_i])
            models_neig_pred.append(model_neig_pred)
            model_pred = []
            for n_i in neigbors_indexes:
                model_pred.append(self.predictions[model_i][n_i])
            models_pred.append(model_pred)
            acc.append(accuracy_score(neigbors_true, model_neig_pred, normalize=True))
            if acc[-1] >self.weights[3]:
                # Adding neighbors predictions
                weights['models_n'].append(int(self.weights[1]/float((1-acc[-1])+0.01)))
                total_pred.extend([pred for j in xrange(weights['models_n'][-1]) for pred in model_pred])
                #print('Predicting Neighbors per sample: %0.4f seconds') % (time.time()-t0)
                # Adding sample prediction
                sample_predictions.append(self.models[self.ind2names[model_i]].predict(x_sample)[0])
                weights['models'].append(int(self.weights[0]/float((1-acc[-1])+0.01))) 
                total_pred.extend([sample_predictions[-1] for j in xrange(weights['models'][-1])])
                total_pred.extend([pred for j in xrange(weights['models'][-1]) for pred in model_neig_pred])
            #print len(x_sample)
            #print self.ind2names[model_i]
            
#                 print 'Model: ' + self.ind2names[model_i] + ' Accuracy: ' + str(accuracy_score(neigbors_true, model_neig_pred, normalize=True))
#                 print 'Predictions'
#                 print model_pred
#                 print 'Representations'
#                 print model_neig_pred
#                 print 'Sample prediction: ' + str(sample_predictions[-1])
        total_pred.extend([n for i, n in enumerate(neigbors_true) for j in xrange(int(weights['true']*(self.k-i)))])
        #print('creating votes: %0.4f seconds') % (time.time()-t0)
        data = Counter(total_pred)
        #data = Counter([k for pred in models_pred for k in pred])
#         print data
#         best_model_ind = acc.index(max(acc))
#         print 'Total pred: ' + str(data.most_common(1)[0][0])
#         print '='*50
        #print len(total_pred)
        #return best_model_ind
        return data.most_common(1)[0][0]

def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %
               (function.func_name, str(t1 - t0))
               )
        return result
    return function_timer


def print_overlaps(predictions, names, verbose=True):
    N = len(names)
    res = numpy.zeros([N,N])
    temp = numpy.zeros([N,N])
    for i in range(0, N):
        for j in range(i+1, N):
            temp[i,j] = len([m for l, m in enumerate(predictions[i]) if (m==predictions[j][l] and m==predictions[N-1][l])])/float(len(predictions[0]))
            res[i,j] = len([(k,v) for k,v in zip(predictions[i], predictions[j]) if k==v])/float(len(predictions[0]))
            if verbose:
                print "%s - %s : %0.3f  overlap | ground-truth coverage: %0.3f" % (names[i],  names[j], 100*res[i,j], 100*temp[i,j])
    return  [res, temp]


@fn_timer
@profile
def main_():
    infolder = "../DATA/pan16-author-profiling-training-dataset-2016-04-25/pan16-author-profiling-training-dataset-english-2016-02-29/"
    outfolder = "models/"
    print('Loading dataset->Grouping User texts.\n')
    dataset = ProfilingDataset(infolder)
    print('Loaded {} users...\n'.format(len(dataset.entries)))
    # get config
    config = dataset.config
    tasks = config.tasks
    print('\n--------------- Thy time of Running ---------------')
    for task in tasks:
        print('Learning to judge %s..' % task)
        # load data
        X, y = dataset.get_data(task)
    X, y = dataset.get_data('age')
    #X, y = dataset.get_data('gender')
    print len(X)
    #print X[0]
    X = preprocess.preprocess(X)
    
    grams3 = TfidfVectorizer(analyzer='word', ngram_range=[2,2], max_features=5000, stop_words='english')
    svm = SVC(kernel='rbf', C=10, gamma=1, class_weight='balanced', probability=False)
    pipe = Pipeline([('3grams',grams3), ('svm', svm)])

    soac = features.SOAC_Model2(max_df=1.0, min_df=1, tokenizer_var='sklearn', max_features=None)
    svm = SVC(kernel='rbf', C=1, gamma=1, class_weight='balanced', probability=False)
    #combined = FeatureUnion([('count_tokens', countTokens), ('count_hash', countHash),
    #                         ('count_urls', countUrls), ('count_replies', countReplies), 
    #                          ('soa', soa), ('soac', soac)])+
    #combined = FeatureUnion([('count_tokens', countTokens), ('count_hash', countHash),
    #                         ('count_urls', countUrls), ('count_replies', countReplies)])
    pipe1 = Pipeline([('soac',soac), ('svm', svm)])

    LSImodel = LSI_Model(num_topics=100)
    svm = SVC(kernel='rbf', C=0.1, gamma=1, class_weight='balanced', probability=False)
    #pipe2 = Pipeline([('counts',combined), ('svm', svm)])
    pipe2 = Pipeline([('LSI',LSImodel), ('svm', svm)])

    # Base Models
    base_models = [pipe, pipe1, pipe2]
    base_model_names = ['3grams', 'soac', 'lsi']

    # Meta Voting Models
    eclf = VotingClassifier(estimators=[("0", pipe), ('1', pipe1), ('2', pipe2)], voting='soft')
    eclfh = VotingClassifier(estimators=[("0", pipe), ('1', pipe1), ('2', pipe2)], voting='hard')
    voting_dic = {'votingf':eclf, 'votingh':eclfh}
    combinator_names = ['majority', 'weights', 'accuracy', 'optimal']
    #meta_models_names = ['votingf', 'votingh', 'space3', 'meta'] + combinator_names
    meta_models_names = ['space3'] + combinator_names
    #meta_models_names = []
    ## all_models ##
    all_models_names = base_model_names + meta_models_names


    #eclf = VotingClassifier(estimators=[("0", pipe), ('1', pipe1)], voting='soft')
    #eclfh = VotingClassifier(estimators=[("0", pipe), ('1', pipe1)], voting='hard')
    #models = [pipe,pipe1,eclf, eclfh]
    #model_names = ['3grams', 'soac', 'voting', 'votingh']

    results = {'over':[]}
    for name in all_models_names:
        results[name] = {'pred': [], 'conf': [], 'rep': [], 'acc': []}

    num_folds = 4
    train_split = 0.3
    meta_split = 0.5
    cv_rounds = 1
    t0 = time.time()
    t1 = t0
    for j in xrange(cv_rounds):
        X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=train_split, stratify=y)
        for i, x in enumerate(X_train):
            if len(x)==0:
                X_train.remove(x)
                y_train.remove(y_train[i])
        for i, x in enumerate(X_cv):
            if len(x)==0:
                X_cv.remove(x)
                y_cv.remove(y_cv[i])
        if meta_split > 0:
            X_meta, X_cv, y_meta, y_cv = train_test_split(X_cv, y_cv, test_size=meta_split, stratify=y_cv)
            print len(X_train), len(X_cv), len(X_meta), len(X_cv) + len(X_train) + len(X_meta), len(X)
        else:
            print len(X_train), len(X_cv), len(X_cv) + len(X_train) , len(X)
        trained_base_models = []
        predictions = []
        base_predictions_cv = []
        base_predictions_meta = []
        for i, model in enumerate(base_models):
            model.fit(X_train,y_train)
            trained_base_models.append(model)
            predict = model.predict(X_cv)
            predictions.append(predict)
            base_predictions_cv.append(predict)
            base_predictions_meta.append(model.predict(X_meta))
            results[base_model_names[i]]['pred'].append(predict)
            results[base_model_names[i]]['acc'].append(accuracy_score(y_cv, predict))
            results[base_model_names[i]]['conf'].append(confusion_matrix(y_cv, predict, labels=list(set(y))))
            results[base_model_names[i]]['rep'].append(classification_report(y_cv, predict, labels=list(set(y))))
        trained_all_models = copy.deepcopy(trained_base_models)
        for name in meta_models_names:
            #print name
            if name =='votingf' or name=='votingh':
                model = voting_dic[name]
                model.fit(X_train, y_train)
                predict = model.predict(X_cv)
            if name == 'space':
                models_for_space = {}
                cv_scores = []
                for i, base_trained_model in enumerate(trained_base_models):
                    models_for_space[base_model_names[i]] = base_trained_model
                    cv_scores.append(base_trained_model.score(X_meta, y_meta))
                model = combinations.SubSpaceEnsemble4_2(models_for_space, cv_scores, k=6, weights=[0.65,0.35,0.32,6], N_rand=10, rand_split=0.6)
                model.fit(X_meta, y_meta)
                predict = model.predict(X_cv)
            if name == 'space3':
                models_for_space = {}
                for i, base_trained_model in enumerate(trained_base_models):
                    models_for_space[base_model_names[i]] = base_trained_model
                model = SubSpaceEnsemble3(models_for_space, k=5, weights= [2,1,3,0.6])
                model.fit(X_train, y_train)
                predict = model.predict(X_cv)
            if name == 'meta':
                model_dic = {}
                for i, base_trained_model in enumerate(trained_base_models):
                    model_dic[base_model_names[i]] = base_trained_model
                model = Metaclassifier(models=model_dic, C=1.0, weights='balanced')
                model.fit(X_meta, y_meta)
                predict = model.predict(X_cv)
            if name in combinator_names:
                #print 'mpike'
                model = combinations.Combinator(scheme=name, weights= [1/float(len(base_predictions_meta)) for i in xrange(len(base_predictions_meta))])
                model.fit(base_predictions_meta, y_meta)
                predict = model.predict(base_predictions_cv)
            trained_all_models.append(model)
            predictions.append(predict)
            results[name]['pred'].append(predict)
            results[name]['acc'].append(accuracy_score(y_cv, predict))
            results[name]['conf'].append(confusion_matrix(y_cv, predict, labels=list(set(y))))
            results[name]['rep'].append(classification_report(y_cv, predict, labels=list(set(y))))
        print('Round %d took: %0.3f seconds') % (j, time.time()-t1)
        t1 = time.time()
    print('Total time: %0.3f seconds') % (time.time()-t0)

    for name in all_models_names:
        print '%%%%%%%%%%%%%%%%  ' + name  + '  % %%%%%%%%%%%%%%%%%%%%%%%'
        print '#################################'
        mean_acc = 0
        mean_prec = 0
        mean_rec = 0
        mean_f1 = 0
        conf = numpy.zeros([5,5])
        for i in xrange(cv_rounds):
            mean_acc += results[name]['acc'][i]
            #print results[key]['report'][i].split('     ')
            mean_prec += float(results[name]['rep'][i].split('     ')[-4][2:])
            mean_rec += float(results[name]['rep'][i].split('     ')[-3][2:])
            mean_f1 += float(results[name]['rep'][i].split('     ')[-2][2:])
            conf += results[name]['conf'][i]
        mean_acc = mean_acc/float(cv_rounds)
        mean_prec = mean_prec/float(cv_rounds)
        mean_rec = mean_rec/float(cv_rounds)
        mean_f1 = mean_f1/float(cv_rounds)
        conf = conf/float(cv_rounds)
        print('Accuracy : {}'.format(mean_acc))
        print('Precision : {}'.format(mean_prec))
        print('Recall : {}'.format(mean_rec))
        print('F1 : {}'.format(mean_f1))
        print('Confusion matrix :\n {}'.format(conf))
        print '#################################'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

if __name__ == '__main__':
    main_()
