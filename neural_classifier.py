from utils import calculate_similarity_single_sent
from classifier_base import BaseClassifier
from sentence_transformers import util, SentenceTransformer
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

class SPECTER_CLS(BaseClassifier):


    def update_field(self, current_data):
        print("Loading and making similarity calculations...")
        self.model_name = 'sentence-transformers/allenai-specter'
        self.model = SentenceTransformer(self.model_name, device='cuda')

        if self.precomputed== "":
            self.emb_source = self.model.encode(list(current_data[self.model_field_name]), device='cuda')  # get our data column
            self.cos_sim = util.pytorch_cos_sim(self.emb_source, self.emb_source)  # .diagonal().tolist()#all similarities
        else:
            print("skipping embedding again")
            self.cos_sim=self.precomputed

        self.was_prepared = True

        # print(self.cos_sim[0][0])

    def train(self):
        """
        There  is no training for a regular expression filter, but the filter could be reset to something else with each training step
        :param reset_filter:
        :param filter:
        :return:
        """
        pass

    def ML_predict(self, current_data):


        print("ML prediction")
        currents = current_data[current_data["discovered_labels"] != ""]

        c_in = currents[currents["discovered_labels"] == 1]
        c_out = currents[currents["discovered_labels"] == 0]
        s = c_in.shape[0] * 3
        if c_out.shape[0] > s:
            c_out = c_out.sample(n=s, random_state=48)
            print(c_out.shape[0])
        currents = pd.concat([c_in, c_out])  # .sample(frac=1, random_state=48)

        print("Predicting. Currently discovered {} labels".format(currents.shape[0]))
        # print('Predicting {} data points using <{}> model'.format(len(self.preprocessed),self.model_name))
        self.predictions = []

        Encoder = LabelEncoder()
        Train_Y = Encoder.fit_transform(currents["discovered_labels"])

        try:
            Train_X_Tfidf = self.Tfidf_vect.transform(currents['backup_processed'])#create vectors for the documents that were previously identified during screening
            All_X_Tfidf = self.Tfidf_vect.transform(current_data["backup_processed"])#vectors for all documants
        except:#this will happen first run only, I know it is a dirty solution but here the TFIDF gets fitted. I need to figure an earlier place to put this, maybe making an init function or so :P
            self.Tfidf_vect = TfidfVectorizer(ngram_range=(1, 3), max_features=75000, min_df=3, strip_accents='unicode')
            self.Tfidf_vect.fit(current_data["backup_processed"])
            Train_X_Tfidf = self.Tfidf_vect.transform(currents['backup_processed'])
            All_X_Tfidf = self.Tfidf_vect.transform(current_data["backup_processed"])


        sgd = SGDClassifier(class_weight="balanced", loss="log_loss")
        parameters = {'alpha': 10.0 ** -np.arange(1, 7)}
        clf = GridSearchCV(sgd, parameters, scoring="roc_auc", cv=StratifiedKFold(n_splits=2))
        clf.fit(Train_X_Tfidf, Train_Y)

        y_preds = clf.predict_proba(All_X_Tfidf)
        probabilities = y_preds[:, 1]
        return probabilities

    def predict(self, current_data):
        print("Using SPECTER for reference priorisation")

        self.retrain_counter+=1#use this to activate ensemble model
        if self.retrain_counter % 5 != 0:#4 out of 5 runs are using the standard AIDOC classifier here
            print("Predicting. Currently discovered {} labels".format(
                len(list(current_data[current_data["discovered_labels"] != ""].index.values))))
            # print('Predicting {} data points using <{}> model'.format(len(self.preprocessed),self.model_name))
            self.predictions = []
            idx = list(current_data[current_data["discovered_labels"] == 1].index.values)  # filter positive labels and get their index values as list
            if len(idx) >= 10:
                idx_pos = random.choices(idx, k=10)
            else:
                idx_pos = idx
            #print("Found {} labels, predicting based on indexes: {}".format(len(idx),idx_pos))  # print first five as sanity check
            #print(idx)

            try:
                sims = calculate_similarity_single_sent(idx_pos, self.cos_sim)#directly calculate similarities. This will fail first time if the embeddings are empty
                sci_title_sim = []
                for data_index in current_data.index.values:
                    sci_title_sim.append(sims[data_index])
                #print(sci_title_sim)
            except:
                #print("Resetting an dpreparing data...")
                self.my_idx = idx_pos
                current_data.reset_index(drop=True, inplace=True)
                self.update_field(current_data)
                sci_title_sim = calculate_similarity_single_sent(self.my_idx, self.cos_sim)
                #print(sci_title_sim)
                #print(current_data.index.values[:10])

            return sci_title_sim
        else:
            return self.ML_predict(current_data)#ensemble-time: use the ML instead :)
