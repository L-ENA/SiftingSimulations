import re

from classifier_base import BaseClassifier

class emptyClassifier(BaseClassifier):

    def train(self):
        """
        There  is no training for a regular expression filter, but the filter could be reset to something else with each training step
        :param reset_filter:
        :param filter:
        :return:
        """
        pass

    def update_field(self):
        pass

    def predict(self, some_data=""):
        #print('Predicting {} data points using <{}> model'.format(len(self.preprocessed),self.model_name))
        self.predictions=[]
        for w in self.preprocessed:
            self.predictions.append(0)#classify all as the same, so no re-ordering happens as all are equal

        return self.predictions


class regexClassifier(BaseClassifier):



    def train(self, filter=r'(\bai\b)|(artificial intelligence)|(machine[\s-]?learn(ing)?)'):
        """
        There  is no training for a regular expression filter, but the filter could be reset to something else with each training step
        :param reset_filter:
        :param filter:
        :return:
        """
        self.filter = filter

    def update_field(self):
        pass

    def predict(self, some_data=""):
        #print('Predicting {} data points using <{}> model'.format(len(self.preprocessed),self.model_name))
        self.predictions = []
        for w in self.preprocessed:
            if re.search(self.filter,w):
                self.predictions.append(1)
            else:
                self.predictions.append(0)
        return self.predictions