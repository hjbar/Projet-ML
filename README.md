# Machine and Deep Learning Data Challenge: Sub-Event Detection

## How to run the code :

### Imports

To execute our program, the following Python libraries must be available : ```sys```, ```os```, ```re```, ```gensim```, ```nltk```, ```numpy```, ```pandas```, ```random```, ```symspellpy```, ```scikit-learn``` and ```xgboost```.

### Location and command

You need to be in the ```code``` directory. Once there, simply run the command ```python3 baseline.py```. 

## Where it expects the original data files :

### The original data files

The ```train_tweets``` and ```eval_tweets``` directories, which contain all the tweets necessary for the project, must be placed in the ```code``` directory.

### The new data files

A new file must be placed in the ```code``` directory. This is the ```en-80k.txt``` file, which is used for spelling correction in tweets. Specifically, it is an English dictionary ordered by word frequency.

## Remarks :

### Predictions

The result of the program, i.e., the prediction, is stored in a file named ```predictions.csv``` at the root of the ```code``` directory.

### Temporary files

To avoid recalculating everything each time, our program saves variables in temporary files. These files are stored in a ```tmp``` directory and another ```tmp_kaggle``` directory.

### Others baselines

In the ```old_codes``` directory, we have included other baselines where we tested different methods and techniques. To execute them, you will need to move them to the root of the ```code``` directory. Additionally, you will need to install some extra libraries, such as ```torch```, ```vaderSentiment```, ```transformers```, and ```tqdm```.