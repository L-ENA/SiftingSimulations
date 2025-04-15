import os
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import re
from sklearn.metrics import classification_report,confusion_matrix
from classifier_base import BaseClassifier
nltk.download('stopwords')
from tqdm import tqdm

# class BaseClassifier:
#     def __init__(self,model_name):
#
#         # self.text_data=data
#         # self.labels=labels
#         self.model_name=model_name
#
#         self.predictions=[]
#         self.preprocessed=[]
#
#
#         self.lemma=WordNetLemmatizer()
#         self.stopwords=stopwords.words('english')
#
#         self.weights_path=self._make_outdir(folder_name="weights")#create output data folders for each specific model
#         self.results_path=self._make_outdir(folder_name="results")
#
#
#
#         # print("Pre=processing {} values".format(len(self.text_data)))
#         # for w in tqdm(self.text_data):
#         #     self.preprocessed.append(self._preprocess(w))
#
#
#
#     def update_data(self, new_preprocessed, new_labels):
#         self.preprocessed=new_preprocessed
#         self.labels=new_labels
#
#     def _make_outdir(self,folder_name="folder"):
#         """
#         Create output directories
#         :param folder_name:
#         :return:
#         """
#         this_path=os.path.join(folder_name,self.model_name)
#
#         if not os.path.exists(this_path):
#             os.mkdir(this_path)
#             return this_path
#
#     def _get_wordnet_pos(self,word):
#         """Map POS tag to first character lemmatize() accepts"""
#         tag = nltk.pos_tag([word])[0][1][0].upper()
#         tag_dict = {"J": wordnet.ADJ,
#                     "N": wordnet.NOUN,
#                     "V": wordnet.VERB,
#                     "R": wordnet.ADV}
#
#         return tag_dict.get(tag, wordnet.NOUN)
#
#     def preprocess(self, word_list):
#         print("Pre=processing {} values".format(len(word_list)))
#         return[self._preprocess(w) for w in word_list]
#
#
#
#     def _preprocess(self,txt):
#         txt=str(txt)
#         lemmatised = [self.lemma.lemmatize(w, self._get_wordnet_pos(w)).lower() for w in nltk.word_tokenize(txt)]
#         values=[w for w in lemmatised if w not in self.stopwords]
#         #print(" ".join(values))
#         return " ".join(values)
#
#     def train(self):
#         raise NotImplementedError("Please Implement this method")
#
#     def predict(self):
#         raise NotImplementedError("Please Implement this method")
#
#     def evaluate(self):
#         print("Confusion matrix:\n{}".format(confusion_matrix(self.labels, self.predictions)))
#         clf=classification_report(self.labels, self.predictions)
#         print("Classification Report:\n{}".format(clf))
#
#     def analyse_predictions(self):
#         pass

class regexClassifier(BaseClassifier):

    def __init__(self, filter=r'(\bai\b)|(artificial intelligence)|(machine[\s-]?learn(ing)?)'):
        self.filter= filter

    def train(self, reset_filter=False, filter=r""):
        """
        There  is no training for a regular expression filter, but the filter could be reset to something else with each training step
        :param reset_filter:
        :param filter:
        :return:
        """
        if reset_filter:
            self.filter = filter

    def predict(self):
        print('Predicting {} data points using <{}> model'.format(len(self.preprocessed),self.model_name))

        for w in self.preprocessed:
            if re.search(self.filter,w):
                self.predictions.append(1)
            else:
                self.predictions.append(0)

class MLClassifiers(BaseClassifier):
    #Quick and dirty, 2 architectures in one. Predict function also does evaluation#


    def train(self):
        """
        There  is no training for a regular expression filter, but the filter could be reset to something else with each training step
        :param reset_filter:
        :param filter:
        :return:
        """
        pass

    def predict(self, current_data):
        import pandas as pd
        import numpy as np
        from nltk.tokenize import word_tokenize
        from nltk import pos_tag
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from sklearn.preprocessing import LabelEncoder
        from collections import defaultdict
        from nltk.corpus import wordnet as wn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn import model_selection, naive_bayes, svm
        from sklearn.linear_model import SGDClassifier
        from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
        from sklearn.metrics import accuracy_score

        currents= current_data[current_data["discovered_labels"] != ""]

        c_in=currents[currents["discovered_labels"] == 1]
        c_out = currents[currents["discovered_labels"] == 0]
        s=c_in.shape[0]*3
        if c_out.shape[0]>s:
            c_out=c_out.sample(n=s, random_state=48)
            print(c_out.shape[0])
        currents=pd.concat([c_in, c_out])#.sample(frac=1, random_state=48)

        print("Predicting. Currently discovered {} labels".format(currents.shape[0]))
        # print('Predicting {} data points using <{}> model'.format(len(self.preprocessed),self.model_name))
        self.predictions = []

        # idx = list(current_data[current_data[
        #                             "discovered_labels"] == 1].index.values)  # filter positive labels and get their index values as list

        ###Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(currents['preprocessed'], currents["discovered_labels"],test_size=0.01,random_state=48)
        Encoder = LabelEncoder()
        Train_Y = Encoder.fit_transform(currents["discovered_labels"])
        #Test_Y = Encoder.fit_transform(Test_Y)


        try:
            Train_X_Tfidf = self.Tfidf_vect.transform(currents['preprocessed'])
            All_X_Tfidf = self.Tfidf_vect.transform(current_data["preprocessed"])
            #print("Without fit")
        except:
            self.Tfidf_vect = TfidfVectorizer(ngram_range=(1, 3), max_features=75000, min_df=3, strip_accents='unicode')
            self.Tfidf_vect.fit(current_data["preprocessed"])
            Train_X_Tfidf = self.Tfidf_vect.transform(currents['preprocessed'])
            All_X_Tfidf = self.Tfidf_vect.transform(current_data["preprocessed"])
        #print(Tfidf_vect.vocabulary_)

        sgd = SGDClassifier(class_weight="balanced", loss="log_loss")
        parameters = {'alpha': 10.0 ** -np.arange(1, 7)}
        clf = GridSearchCV(sgd, parameters, scoring="roc_auc", cv=StratifiedKFold(n_splits=2))
        clf.fit(Train_X_Tfidf, Train_Y)

        y_preds = clf.predict_proba(All_X_Tfidf)
        probabilities = y_preds[:, 1]
        return probabilities


        # fit the training dataset on the NB classifier
        # Naive = naive_bayes.MultinomialNB()
        # Naive.fit(Train_X_Tfidf, Train_Y)
        # # predict the labels on validation dataset
        # predictions_NB = Naive.predict(Test_X_Tfidf)
        #
        # print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, Test_Y) * 100)
        # clf = classification_report(Test_Y, predictions_NB)
        # print("Classification Report:\n{}".format(clf))
        #
        #
        # SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        # SVM.fit(Train_X_Tfidf, Train_Y)
        # # predict the labels on validation dataset
        # predictions_SVM = SVM.predict(Test_X_Tfidf)
        # # Use accuracy_score function to get the accuracy
        # print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)
        # clf = classification_report(Test_Y, predictions_SVM)
        # print("Classification Report:\n{}".format(clf))




