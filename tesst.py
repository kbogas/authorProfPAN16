#!/usr/bin/python

import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pan import ProfilingDataset,createDocProfiles,create_target_prof_trainset
from tictacs import from_recipe
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.cross_validation import train_test_split

log = []


def test_data(dataset, task, model, split=0.3):
    """ train and cross validate a model

    :lang: the language
    :task: the task we want to classify for , ex: age

    """

    # if (task != "age") and (task != "gender"):
    #     X, y = dataset.get_data(task)
    # else:
    #     docs = createDocProfiles(dataset)
    #     X, y = create_target_prof_trainset(docs, task)
    X, y = dataset.get_data(task)
    #docs = createDocProfiles(dataset)
    #X, y = create_target_prof_trainset(docs, task)
    # y = [yy.lower() for yy in y]
    # get parameters for grid search if it exists - else pass empty dict
    from collections import Counter
    import pprint
    pprint.pprint(Counter(y))
    # print(sorted(list(set(y))))
    #pprint.pprint(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42, stratify=y)
    log.append('splitted dataset. Train: %s, Test: %s' % (len(X_train), len(X_test)))
    # tictac = from_recipe(config.recipes[task])
    # all_models[task] = tictac.fit(X_train, y_train)
    model.fit(X_train, y_train)
    log.append('Trained model')
    predict = model.predict(X_test)
    log.append('Tested model')
    log.append('\n-- Predictions for %s --' % task)
    try:
        # if it's classification we measure micro and macro scores
        acc = accuracy_score(y_test, predict)
        conf = confusion_matrix(y_test, predict, labels=sorted(list(set(y_test))))
        all_m = precision_recall_fscore_support(y_test, predict)
        log.append('Metrics: \n')
        log.append('Accuracy : {}\n'.format(acc))
        log.append('Precision : {}\n'.format(all_m[0]))
        log.append('Recall : {}\n'.format(all_m[1]))
        log.append('Fbeta Score : {}\n'.format(all_m[2]))
        log.append('Support : {}\n'.format(all_m[3]))
        log.append('Labels : {}\n'.format(sorted(list(set(y)))))
        log.append('Confusion matrix :\n {}'.format(conf))
    except ValueError:
        print "Shouldn't be here!"

if __name__ == '__main__':
    parser = ArgumentParser(description='Train a model with crossvalidation'
                            ' on pan dataset - used for testing purposes ', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', type=str,
                        required=True, dest='infolder',
                        help='path to folder with pan dataset for a language')
    parser.add_argument('-s', '--split', type=float,
                        dest='split', default=0.3,
                        help='Test set percentage.')

    args = parser.parse_args()
    infolder = args.infolder
    split = args.split
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
        test_data(dataset, task, tictac, split)
    # print results at end
    print('\n--------------- Thy time of Judgement ---------------')
    print ('Time: {} seconds.\n'.format(str(time.time()-time_start)))
    for message in log:
        print(message)
