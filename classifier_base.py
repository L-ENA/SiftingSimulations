import os
import nltk
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import re
from sklearn.metrics import classification_report,confusion_matrix
#!pip install pyyaml==5.4.1


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from tqdm import tqdm

class BaseClassifier:
    def __init__(self,model_name, precomputed=""):

        # self.text_data=data
        # self.labels=labels
        self.retrain_counter = 0
        self.model_name=model_name
        self.precomputed=precomputed

        self.predictions=[]
        self.preprocessed=[]
        self.labels=[]


        self.lemma=WordNetLemmatizer()
        self.stopwords=stopwords.words('english')

        self.weights_path=self._make_outdir(folder_name="weights")#create output data folders for each specific model
        self.results_path=self._make_outdir(folder_name="results")
        self.model_field_name=""


        # print("Pre=processing {} values".format(len(self.text_data)))
        # for w in tqdm(self.text_data):
        #     self.preprocessed.append(self._preprocess(w))


    def set_model_field(self, field_name):

        print("Set model field name to: {}".format(field_name))
        self.model_field_name=field_name

    def update_data(self, new_preprocessed):
        self.preprocessed=new_preprocessed



    def _make_outdir(self,folder_name="folder"):
        """
        Create output directories
        :param folder_name:
        :return:
        """
        this_path=os.path.join(folder_name,self.model_name)

        if not os.path.exists(this_path):
            os.mkdir(this_path)
            return this_path

    def _get_wordnet_pos(self,word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    def preprocess(self, word_list):
        from sklearn.feature_extraction.text import TfidfVectorizer

        print("Pre-processing {} values".format(len(word_list)))
        values=[]
        for w in tqdm(word_list):
             values.append(self._preprocess(w))

        # Tfidf_vect = TfidfVectorizer(ngram_range=(1, 3), max_features=75000, min_df=3, strip_accents='unicode')
        #
        #
        # #Tfidf_vect.fit(values)
        # vects = Tfidf_vect.fit_transform(values)
        # print(vects.shape)

        return values



    def _preprocess(self,txt):
        txt=str(txt)
        #lemmatised = [self.lemma.lemmatize(w, self._get_wordnet_pos(w)).lower() for w in nltk.word_tokenize(txt)]
        lemmatised = [w.lower() for w in nltk.word_tokenize(txt)]
        values=[w for w in lemmatised if w not in self.stopwords]
        #print(" ".join(values))
        return " ".join(values)

    def train(self):
        raise NotImplementedError("Please Implement this method")

    def predict(self):
        raise NotImplementedError("Please Implement this method")

    def update_field(self):
        raise NotImplementedError("Please Implement this method")


    def evaluate(self):
        print("Confusion matrix:\n{}".format(confusion_matrix(self.labels, self.predictions)))
        clf=classification_report(self.labels, self.predictions)
        print("Classification Report:\n{}".format(clf))

    def analyse_predictions(self):
        pass