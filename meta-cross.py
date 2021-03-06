#!/usr/bin/python

import time
from argparse import ArgumentParser
from tictacs import from_recipe
from pan import ProfilingDataset
from sklearn.ensemble import VotingClassifier
from sklearn.grid_search import GridSearchCV
from json import dumps
from pan.features import Metaclassifier
from sklearn.cross_validation import train_test_split
# import dill
# import cPickle as pickle
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
log = []


def test_models(name, model, X, y):

    accuracy = model.score(X, y)
    log.append('Model %s  with Accuracy : %s' % (name, accuracy))
    print 'Model %s  with Accuracy : %s' % (name, accuracy)
    return

def cross_val(X, y, dataset, task, model, num_folds=4):
    """ train and cross validate a model

    :lang: the language
    :task: the task we want to classify for , ex: age

    """

    # if (task != "age") and (task !="gender"):
    #    X, y = dataset.get_data(task)
    # else:
    #    docs = createDocProfiles(dataset)
    #    X, y = create_target_prof_trainset(docs, task)
    # docs = createDocProfiles(dataset)
    # X, y = create_target_prof_trainset(docs, task)
    #X, y = dataset.get_data(task)
    # y = [yy.lower() for yy in y]
    # get parameters for grid search if it exists - else pass empty dict
    params = model.grid_params if hasattr(model, 'grid_params') else dict()
    #print params
    from collections import Counter
    import pprint
    print "Num of samples: " + str(len(y))
    pprint.pprint(Counter(y))
    print '\nCreating model for %s - %s' % (dataset.lang, task)
    print 'Trainining instances: %s\n' % (len(X))
    print 'Using %s fold validation' % (num_folds)
    # get data
    log.append('\nResults for %s - %s with classifier %s' %
               (dataset.lang, task, model.__class__.__name__))
    if task in dataset.config.classifier_list:
        grid_cv = GridSearchCV(model, params, cv=num_folds, verbose=1,
                               n_jobs=-1, refit=True)
        grid_cv.fit(X, y)
        # y_pred = grid_cv.best_estimator_.predict(X)
        # pprint.pprint(y_pred)
        # pprint.pprint(y)
        # conf = confusion_matrix(y, y_pred, labels=list(set(y)))
        accuracy = grid_cv.best_score_
        # accuracy2 = accuracy_score(y, y_pred)
        log.append('best params: %s' % grid_cv.best_params_)
        log.append('Accuracy mean : %s' % accuracy)
        import pprint
        pprint.pprint(grid_cv.grid_scores_)
        log.append('Model for task : %s' % task)
        log.append(grid_cv.grid_scores_)
        log.append('\n')
        with open('./comb_res/res.txt', 'a') as out:
            out.write('Results: %s - %s, params: %s ,Accuracy_Mean: %s\n' %
                      (dataset.lang, task,
                       dumps(grid_cv.best_params_), grid_cv.best_score_))
        return grid_cv.best_estimator_


if __name__ == '__main__':
    parser = ArgumentParser(description='Train a model with crossvalidation'
                            ' on pan dataset - used for testing purposes ')
    parser.add_argument('-i', '--input', type=str,
                        required=True, dest='infolder',
                        help='path to folder with pan dataset for a language')
    parser.add_argument('-n', '--numfolds', type=int,
                        dest='num_folds', default=4,
                        help='Number of folds to use in cross validation')

    args = parser.parse_args()
    infolder = args.infolder
    num_folds = args.num_folds
    time_start = time.time()
    split = 0.3
    print('Loading dataset->Grouping User texts.\n')
    dataset = ProfilingDataset(infolder)
    print('Loaded {} users...\n'.format(len(dataset.entries)))
    # get config
    config = dataset.config
    tasks = config.tasks
    print('\n--------------- Thy time of Running ---------------')
    list_model_names = ['tictac', 'soac', 'lda', 'voting', 'meta']
    total_model = {}
    for model_name in list_model_names:
        all_models = {}
        if model_name != 'voting' and model_name != 'meta':
            for task in tasks:
                print('Learning to judge %s with %s' % (task, model_name))
                # load data
                X, y = dataset.get_data(task)
                if 'meta' in list_model_names:
                    X, X_cv, y, y_cv = train_test_split(X, y, test_size=split, random_state=42, stratify=y)
                tictac = from_recipe(config.recipes[task + '-' + model_name])
                outline = ""
                for step in tictac.steps:
                    if step[0] == "features":
                        # print type(step[1])
                            for tf in step[1].transformer_list:
                                # print type(tf[1])
                                # print type(tf[1].get_params())
                                outline += tf[0] + " with Params:[" + str(tf[1].get_params()) + "]+"
                    else:
                        # if hasattr(step[1], 'get_params'):
                            # outline += step[0] + " with Params:[" + str(step[1].get_params()) + "]+"
                        # else:
                            # outline += step[0]+ "+"
                        #outline += step[0] + "+"
                        outline += step[0] +" with Params:[" + str(step[1].get_params()) + "]+"
                outline = outline[:-1] + "\n"
                print('Task:{}, Pipeline:{}'.format(task, outline))
                #all_models[task] = tictac
                all_models[task] = cross_val(X, y, dataset, task, tictac, num_folds)
        elif model_name == 'voting':
            for task in tasks:
                model_list = []
                print('Learning to judge %s with %s' % (task, model_name))
                #X, y = dataset.get_data(task)
                X, y = dataset.get_data(task)
                if 'meta' in list_model_names:
                    X, X_cv, y, y_cv = train_test_split(X, y, test_size=split, random_state=42, stratify=y)
                model_list = [(name, all_model[task])for name, all_model in total_model.iteritems() if (name != 'voting' and name != 'meta')]
                all_models[task] = cross_val(X, y, dataset, task, VotingClassifier(estimators=model_list, voting='soft'), num_folds)
        elif model_name == 'meta':
            for task in tasks:
                model_dic = {}
                for name, all_model in total_model.iteritems():
                    if (name != 'voting' and name != 'meta'):
                        model_dic[name] = all_model[task]
                print('Learning to judge %s with %s' % (task, model_name))
                Meta = Metaclassifier(models=model_dic, C=1.0, weights='balanced')
                X, y = dataset.get_data(task)
                X, X_cv, y, y_cv = train_test_split(X, y, test_size=split, random_state=42, stratify=y)
                X_cv, X_test, y_cv, y_test = train_test_split(X_cv, y_cv, test_size=0.5, random_state=42, stratify=y_cv)
                print "Len X: " + str(len(X))
                print "Len X_cv: " + str(len(X_cv))
                print "Len X_test: " + str(len(X_test))
                #params = {'C': [0.01, 0.1, 1, 10, 100, 1000]}
                #from sklearn import grid_search
                #clf = grid_search.GridSearchCV(Meta, params)
                #clf.fit(X_cv, y_cv)
                Meta.fit(X_cv, y_cv)
                #model_dic['meta'] = Meta
                import pprint
                print "Meta coeff:"
                pprint.pprint(Meta.svc.coef_)
                print "Meta intercept"
                pprint.pprint(Meta.svc.intercept_)
                for name in model_dic.keys():
                    test_models(name, model_dic[name], X_test, y_test)
                print "Models"
                print Meta.models.keys()
                #test_models('Meta', clf, X_test, y_test)
                test_models('Meta', Meta, X_test, y_test)
                test_models('voting', total_model['voting'][task], X_test, y_test)
                #all_models[task] = cross_val(X_test, y_test, dataset, task, Meta, num_folds)
        else:
            print("Can't do anything with this %s model!" % model_name)
        print("Tha swsw to modelo %s!!" % model_name)
        total_model[model_name] = all_models
    print('\n--------------- Thy time of Judgement ---------------')
    print ('Time: {} seconds.\n'.format(str(time.time()-time_start)))
    with open('./comb_res/res.txt', 'a') as out:
            out.write('Time: {} seconds.\n'.format(str(time.time()-time_start)))
    for message in log:
        print(message)


            # elif model_name == 'meta':
        #     for task in tasks:
        #         model_preds = [all_model[task].predict(X_cv) for all_model in total_model.values()]
        #         print('Learning to judge %s with %s' % (task, model_name))
        #         pred_features = model_preds[0]
        #         for model_pred in model_preds[1:]:
        #             pred_features = vstack((pred_features, model_pred))