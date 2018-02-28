"""Template for assignment 7. (machine-learning)."""

import glob
import re
from string import punctuation as punct  # string of common punctuation chars

import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
import numpy as np
import pandas
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# import model classes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# TODO change to the location of your Mini-CORE corpus
MC_DIR = '/Users/Kyle/desktop/mini-core-project/Mini-CORE/'
words_only = RegexpTokenizer(r'\w+').tokenize
LINKRE = (r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|('
          r'?:%[0-9a-fA-F][0-9a-fA-F]))+')


def cleansificate(file):
    """
    Purification/baptism of the file in question to prepare it for tokenization
    e.g. strips HTML and anything in a tag
    """
    cleansificated_str = ''
    # with open(file, 'r') as f:
    text = file.read()
    # removes any XML tags
    no_html = re.sub('<[^<]+?>', '', text)
    # removes any links
    cleansificated_str = re.sub(LINKRE, '', no_html)
    return cleansificated_str


def subcorp(name):
    """Extract subcorpus from filename.
    name -- filename
    The subcorpus is the first abbreviation after `1+`.
    """
    return name.split('+')[1]


def ttr(in_Text):
    """Compute type-token ratio for input Text.
    in_Text -- nltk.Text object or list of strings
    """
    return len(set(in_Text)) / len(in_Text)


def cttr(raw_text):
    """corrected ttr which handles varying sizes
    raw_text -- un-tokenized
    """
    lc_tokens = words_only(raw_text.lower())
    unique_tkns = set(lc_tokens)
    if len(lc_tokens) != 0:
        cttr = len(unique_tkns) / np.sqrt(2*len(lc_tokens))
    else:
        cttr = 0
    return cttr


def get_function_words():
    """
    Gets all the function words stored in the local file 'function_words.txt'
    """
    function_word_set = None
    with open('function_words.txt', 'r') as f:
        f = f.read()
        function_word_set = set(word_tokenize(f.lower()))
    return function_word_set


def fnct_per_tok(in_Text):
    """Compute the number of function words per adjusted token in the text
    in_Text is the tokenized text.
    """
    function_words = get_function_words()
    total_tokens = len(in_Text)
    total_fncts = 0
    for word in in_Text:
        if word in function_words:
            total_fncts += 1
    return total_fncts / total_tokens


def song_words(in_Text):
    """Calculates the number of 'song words' are said in a text
    """
    song_words = set(['la', 'oh', 'na', 'lyrics', 'song',
                      'sing', 'gon', 'chorus', 'sheeran', 'like'])
    count = 0
    total_len = len(in_Text)
    for word in in_Text:
        if word in song_words:
            count += 1
    return (count / total_len) * 100


def IN_words(in_Text):
    """a list of frequent words in the wikipedia register
    """

    in_words = set(['states', 'time', 'used', 'state', 'united',
                    'war', 'government', 'u.s.', 'american', 'national'])
    count = 0
    total_len = len(in_Text)
    for word in in_Text:
        if word in in_words:
            count += 1
    return (count / total_len) * 100


def ip_words(in_Text):
    """ a list of frequent words appearing in informational persuasion
    """
    ip_words = set(['book', 'nov', 'pm', 'read', 'molly', 'pre-order', 'life',
                    'trial'])
    count = 0
    total_len = len(in_Text)
    for word in ip_words:
        if word in ip_words:
            count += 1
    return (count / total_len) * 100


def news_words(in_Text):
    """list of news words"""
    news_words = set(['said', 'obama', 'mr', 'president', 'years',
                      'know', 'really'])
    count = 0
    total_len = len(in_Text)
    for word in in_Text:
        if word in news_words:
            count += 1
    return (count / total_len) * 100


def interview_words(in_Text):
    """ a list of interview words! YAY! """
    inter_words = set(['really', 'going', 'well', 'like'])
    count = 0
    total_len = len(inter_words)
    for word in in_Text:
        if word in inter_words:
            count += 1
    return (count / total_len) * 100


# add feature names HERE
feat_names = ['cttr', 'song_words', 'IN_words', 'ip_words', 'news_words',
              'interview_words', 'genre']
with open('mc_feat_names.txt', 'w') as name_file:
    name_file.write('\t'.join(feat_names))

with open('mc_features.csv', 'w') as out_file:
    for f in glob.glob(MC_DIR + '*.txt'):
        print('.', end='', flush=True)  # show progress; print 1 dot per file
        with open(f) as the_file:
            raw_text = cleansificate(the_file)
        tok_text = nltk.word_tokenize(raw_text)
        # call the function HERE
        print(cttr(raw_text), song_words(tok_text), IN_words(tok_text),
              ip_words(tok_text), news_words(tok_text),
              interview_words(tok_text), subcorp(f),
              sep=',', file=out_file)
    print()  # newline after progress dots

###############################################################################
# Do not change anything below this line! The assignment is simply to try to
# design useful features for the task by writing functions to extract those
# features. Simply write new functions and add a label to feat_names and call
# the function in the `print` function above that writes to out_file. MAKE SURE
# TO KEEP the order the same between feat_names and the print function, ALWAYS
# KEEPING `'genre'` AND `subcorp(f)` AS THE LAST ITEM!!

###############################################################################
# Load dataset
with open('mc_feat_names.txt') as name_file:
    names = name_file.read().strip().split('\t')
len_names = len(names)
with open('mc_features.csv') as mc_file:
    dataset = pandas.read_csv(mc_file, names=names,  # pandas DataFrame object
                              keep_default_na=False, na_values=['_'])  # avoid 'NA' category being interpreted as missing data  # noqa
print(type(dataset))

# Summarize the data
print('"Shape" of dataset:', dataset.shape,
      '({} instances of {} attributes)'.format(*dataset.shape))
print()
print('"head" of data:\n', dataset.head(20))  # head() is a method of DataFrame
print()
print('Description of data:\n:', dataset.describe())
print()
print('Class distribution:\n', dataset.groupby('genre').size())
print()

# Visualize the data
print('Drawing boxplot...')
grid_size = 0
while grid_size ** 2 < len_names:
    grid_size += 1
dataset.plot(kind='box', subplots=True, layout=(grid_size, grid_size),
             sharex=False, sharey=False)
fig = plt.gcf()  # get current figure
fig.savefig('boxplots.png')

# histograms
print('Drawing histograms...')
dataset.hist()
fig = plt.gcf()
fig.savefig('histograms.png')

# scatter plot matrix
print('Drawing scatterplot matrix...')
scatter_matrix(dataset)
fig = plt.gcf()
fig.savefig('scatter_matrix.png')
print()

print('Splitting training/development set and validation set...')
# Split-out validation dataset
array = dataset.values  # numpy array
feats = array[:,0:len_names - 1]  # to understand comma, see url in next line:
labels = array[:,-1]  # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
print('\tfull original data ([:5]) and their respective labels:')
print(feats[:5], labels[:5], sep='\n\n', end='\n\n\n')
validation_size = 0.20
seed = 7
feats_train, feats_validation, labels_train, labels_validation = model_selection.train_test_split(feats, labels, test_size=validation_size, random_state=seed)
# print('\ttraining data:\n', feats_train[:5],
#       '\ttraining labels:\n', labels_train[:5],
#       '\tvalidation data:\n', feats_validation[:5],
#       '\tvalidation labels:\n', labels_validation[:5], sep='\n\n')

# Test options and evaluation metric
seed = 7  # seeds the randomizer so that 'random' choices are the same in each run
scoring = 'accuracy'
print()

print('Initializing models...')
# Spot Check Algorithms
models = [('LR', LogisticRegression()),
          ('LDA', LinearDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('NB', GaussianNB()),
          ('SVM', SVC())]
print('Training and testing each model using 10-fold cross-validation...')
# evaluate each model in turn
results = []
names = []
for name, model in models:
    # https://chrisjmccormick.files.wordpress.com/2013/07/10_fold_cv.png
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, feats_train, labels_train,
                                                 cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = '{}: {} ({})'.format(name, cv_results.mean(), cv_results.std())
    print(msg)
print()

print('Drawing algorithm comparison boxplots...')
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
fig = plt.gcf()
fig.savefig('compare_algorithms.png')
print()

# Make predictions on validation dataset
best_model = KNeighborsClassifier()
best_model.fit(feats_train, labels_train)
predictions = best_model.predict(feats_validation)
print('Accuracy:', accuracy_score(labels_validation, predictions))
print()
# print('Confusion matrix:')
# cm_labels = 'Iris-setosa Iris-versicolor Iris-virginica'.split()
# print('labels:', cm_labels)
# print(confusion_matrix(labels_validation, predictions, labels=cm_labels))
# print()
print('Classification report:')
print(classification_report(labels_validation, predictions))
