#!/usr/bin/python

import os
from argparse import ArgumentParser
from sklearn.externals import joblib
from tictacs import from_recipe
from pan import ProfilingDataset
# import dill
#import cPickle as pickle
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
    all_models = {}
    for task in tasks:
        print('Learning to judge %s..' % task)
        # load data
        X, y = dataset.get_data(task)
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
        all_models[task] = tictac.fit(X, y)
    modelfile = os.path.join(outfolder, '%s.bin' % dataset.lang)
    print('Writing model to {}'.format(modelfile))
    #fo = open(modelfile,  'wb')
    #import pprint
    #print type(all_models)
    #print modelfile
    #dill.dump(all_models, fo, protocol=pickle.HIGHEST_PROTOCOL)
    #fo.close()
    #pickle.dump(all_models, modelfile)
    # dill.dump(all_models, modelfile)
    joblib.dump(all_models, modelfile, compress=3)
