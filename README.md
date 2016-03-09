# NCSR Demokritos submission to Pan 2016.
This work is  based on last year's submission for [PAN15](https://github.com/iit-Demokritos/pangram)..

## Installation:

### Dataset:
In order to run the examples you will need to download the corpus for the author profiling task
from the PAN website:

http://pan.webis.de/clef16/pan16-web/author-profiling.html

### Requirements:

Install the requirements 

pip install -r requirements.txt

### Module:

You can also install the module if you would like to check it out from ipython.
git clone this project
cd projectfolder
pip install --user .



Package consists of a python module and scripts for:
- crossvalidating
- training
- testing
models on the PAN 2016 dataset.

## Example usage:

- python tesst.py -i pan16-author-profiling-training-dataset/pan16-author-profiling-training-dataset-english/
- python cross.py -i pan16-author-profiling-training-dataset/pan16-author-profiling-training-dataset-english/


## Configuration:
Configuration follows the same conventions used for[PAN15 submission](https://github.com/iit-Demokritos/pangram).
In the config folder is a toy setup of the configuration for pangram. It is based on the
[YAML](http://yaml.org) format.

Settings currently configurable are:
- Pan dataset settings for each language
- Feature groupings, preprocessing for each feature group, and classifier settings

In config/languages there is a file for each language which specifies where each attribute
to be predicted is in the truth file that contains the label for the training set. For each
of these attributes, you can set a file that contains the feature grouping and preprocessing
settings. In the example provided the mapping is the same for each language, but this need
not be the case.

In config/features the settings for each feature group can be found. The format is in the form
label of:
> label of feature group
>  - feature extractor 1
>  - feature extractor 2
>  - ..
>  preprocessing :
>    label: label this so that it doesn't get computed twice if it has been defined elsewhere
>    pipe: 
>        - method 1
>        - method 2
>        - ...
In the above snippet, feature extractor names are expected to be defined in pan/features.py.
Similarly, the above methods are expected to be defined in pan/preprocess.py and process a mutable iterable in place. (in our case a list of texts)

## License
Pangram - NCSR Demokritos submission for Pan 2016
Copyright 2016 Konstantinos Bougiatiotis

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======
# pangram
Author Profiling module
