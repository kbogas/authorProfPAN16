#!/usr/bin/python

import os
from argparse import ArgumentParser
from numpy import vstack
from sklearn.externals import joblib
from tictacs import from_recipe
from pan import ProfilingDataset
from sklearn.ensemble import VotingClassifier
# import dill
# import cPickle as pickle
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix


if __name__ == '__main__':
    parser = ArgumentParser(description='Train pan model on pan dataset')
    parser.add_argument('-i', '--input', type=str,
                        required=True, dest='infolder',
                        help='path to folder with pan dataset for a language')
    parser.add_argument('-o', '--output', type=str,
                        required=True, dest='outfolder',
                        help='path to folder where model should be written')

    args = parser.parse_args()
    infolder = args.infolder
    outfolder = args.outfolder
    print('Loading dataset->Grouping User texts.\n')
    dataset = ProfilingDataset(infolder)
    print('Loaded {} users...\n'.format(len(dataset.entries)))
    # get config
    config = dataset.config
    tasks = config.tasks
    print('\n--------------- Thy time of Running ---------------')
    list_model_names = ['tictac', 'lda', 'voting']
    total_model = {}
    for model_name in list_model_names:
        all_models = {}
        if model_name != 'voting':
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
                        outline += step[0] + "+"
                outline = outline[:-1] + "\n"
                print('Task:{}, Pipeline:{}'.format(task, outline))
                all_models[task] = tictac.fit(X, y)
        elif model_name == 'voting':
            for task in tasks:
                model_list = []
                print('Learning to judge %s with %s' % (task, model_name))
                X, y = dataset.get_data(task)
                if 'meta' in list_model_names:
                    X, X_cv, y, y_cv = train_test_split(X, y, test_size=split, random_state=42, stratify=y)
                model_list = [(name, all_model[task])for name, all_model in total_model.iteritems() if name != 'ensemble']
                all_models[task] = VotingClassifier(estimators=model_list, voting='hard').fit(X, y)
        else:
            print("Can't do anything with this %s model!" % model_name)
        total_model[model_name] = all_models
    modelfile = os.path.join(outfolder, '%s-total.bin' % dataset.lang)
    print('Writing model to {}'.format(modelfile))
    # dill.dump(all_models, modelfile)
    joblib.dump(total_model, modelfile, compress=3)


            # elif model_name == 'meta':
        #     for task in tasks:
        #         model_preds = [all_model[task].predict(X_cv) for all_model in total_model.values()]
        #         print('Learning to judge %s with %s' % (task, model_name))
        #         pred_features = model_preds[0]
        #         for model_pred in model_preds[1:]:
        #             pred_features = vstack((pred_features, model_pred))