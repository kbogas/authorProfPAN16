#!/usr/bin/python

import time
from argparse import ArgumentParser
from pan import ProfilingDataset, createDocProfiles, create_target_prof_trainset
from tictacs import from_recipe
from json import dumps
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score

log = []


def cross_val(dataset, task, model, num_folds=4):
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
    X, y = dataset.get_data(task)
    # y = [yy.lower() for yy in y]
    # get parameters for grid search if it exists - else pass empty dict
    params = model.grid_params if hasattr(model, 'grid_params') else dict()
    # from collections import Counter
    # import pprint
    # pprint.pprint(Counter(y))
    print '\nCreating model for %s - %s' % (dataset.lang, task)
    print 'Trainining instances: %s\n' % (len(X))
    print 'Using %s fold validation' % (num_folds)
    # get data
    log.append('\nResults for %s - %s with classifier %s' %
               (dataset.lang, task, model.__class__.__name__))
    if task in dataset.config.classifier_list:
        grid_cv = GridSearchCV(model, params, cv=num_folds, verbose=1,
                               n_jobs=1)
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
        with open('./comb_res/res.txt', 'a') as out:
            out.write('Results: %s - %s, params: %s ,Accuracy_Mean: %s\n' %
                      (dataset.lang, task,
                       dumps(grid_cv.best_params_), grid_cv.best_score_))
        # log.append('Best accuracy: {} '.format(accuracy2))
        # log.append('Best Confusion matrix :\n {}'.format(conf))
    else:
        # if it's not, we measure mean square root error (regression)
        raise KeyError('task %s was not found in task list!' % task)

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
    print('Loading dataset...')
    dataset = ProfilingDataset(infolder)
    print('Loaded %s users...\n' % len(dataset.entries))
    config = dataset.config
    tasks = config.tasks
    print('\n--------------- Thy time of Running ---------------')
    for task in tasks:
        tictac = from_recipe(config.recipes[task])
        outline = ""
        for step in tictac.steps:
            if step[0]=="features":
                for tf in step[1].transformer_list:
                    outline += tf[0] + "+"
            else:
                outline += step[0]+ "+"
        outline = outline[:-1]
        print('Task:{}, Pipeline:{}'.format(task, outline))
        with open('./comb_res/res.txt', 'a') as out:
            out.write('Task:{}, Pipeline:{}'.format(task, outline))
        cross_val(dataset, task, tictac, num_folds)
    # print results at end
    print('\n--------------- Thy time of Judgement ---------------')
    print ('Time: {} seconds.\n'.format(str(time.time()-time_start)))
    with open('./comb_res/res.txt', 'a') as out:
            out.write('Time: {} seconds.\n'.format(str(time.time()-time_start)))
    for message in log:
        print(message)
