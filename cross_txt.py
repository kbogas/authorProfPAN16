#!/usr/bin/python

from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
from sklearn.cross_validation import train_test_split
from tictacs import from_recipe
from sklearn.grid_search import GridSearchCV
import pprint


def cross_val(X, y, model, num_folds=4):

    params = model.grid_params if hasattr(model, 'grid_params') else dict()
    print params
    print model
    # from collections import Counter
    # import pprint
    # pprint.pprint(Counter(y))
    print 'Trainining instances: %s\n' % (len(X))
    print 'Using %s fold validation' % (num_folds)
    # get data
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    grid_cv = GridSearchCV(model, params, cv=num_folds, verbose=1,
                           n_jobs=-1, refit=False)
    #grid_cv.fit(X_train, y_train)
    grid_cv.fit(X, y)
    print('GridSearch Params')
    pprint.pprint(grid_cv.grid_scores_)
    print('Best params')
    pprint.pprint(grid_cv.best_params_)
    print('Best score')
    pprint.pprint(grid_cv.best_score_)
    # predict = grid_cv.best_estimator_.predict(X_test)
    # acc = accuracy_score(y_test, predict)
    # conf = confusion_matrix(y_test, predict, labels=sorted(list(set(y_test))))
    # rep = classification_report(y_test, predict, target_names=sorted(list(set(y_test))))
    # print('Accuracy : {}'.format(acc))
    # print('Confusion matrix :\n {}'.format(conf))
    # print('Classification report :\n {}'.format(rep))

    
    # log.append('Best accuracy: {} '.format(accuracy2))
    # log.append('Best Confusion matrix :\n {}'.format(conf))




if __name__ == '__main__':
    parser = ArgumentParser(description='Test trained model on pan dataset')
    parser.add_argument('-x', '--x_path', type=str,
                        required=True, dest='x_path',
                        help='path to folder with pan dataset for a language')
    parser.add_argument('-y', '--y_path', type=str,
                        required=True, dest='y_path',
                        help='path to folder where results should be written')
    parser.add_argument('-n', '--numfolds', type=int,
                        dest='num_folds', default=4,
                        help='Number of folds to use in cross validation')

    args = parser.parse_args()
    X_path = args.x_path
    y_path = args.y_path
    num_folds = args.num_folds
    # This part for tira-io
    with open(X_path, 'r') as xin:
        X = pickle.load(xin)
    with open(y_path, 'r') as yin:
        y = pickle.load(yin)
    ######
    print('Number of docs: %d,%d' % (len(X), len(y)))
    for task in ['gender']:
        tictac = from_recipe("./config/recipes/gender.yml")
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
                outline += step[0] + "+"
        outline = outline[:-1] + "\n"
        print('Task:{}, Pipeline:{}'.format(task, outline))
        cross_val(X, y, tictac, num_folds)
