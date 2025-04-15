import pandas as pd
import plotly.express as px

class ActiveLearner:
    def __init__(self,classifier,data,field="ScientificTitle", time_to_retrain=10,model_name="regex", do_preprocess=True, precomputed=""):
        """
        Taking a classifier instance from one of the classifer architectures in classifiers.py, and a dataset
        :param classifier:
        :param data:
        """

        ##############Gold standard (or non-labelled) data
        self.precomputed=precomputed
        self.all_data=data
        self.all_data["discovered_labels"]=""#the screened references and discovered gold-standard labels
        self.all_data["predictions"] = 0
        #self.all_data= self.all_data.sample(frac=1, random_state=42)
        self.field=field[:500]

        ########################plotting and eval
        self.precisions=[]
        self.recalls=[]
        self.f1s=[]
        self.num_found_list=[]#number of relevant references found at each step
        self.num_steps=0
        self.step_list=[]#number of references screened
        ###################################The classifier with all methods overwritten as needed
        self.do_preprocess=do_preprocess
        self.stop_reordering=False
        self.classifier=classifier

        self.time_to_retrain=time_to_retrain
        self.classifier=classifier(model_name, precomputed=precomputed)
        self.change_field()
        self.discovery_order=[]
        print("Done setting up classifier")

    def change_field(self):
        print("Setting {} as classification field.".format(self.field))
        if self.do_preprocess:
            self.all_data["preprocessed"] = self.classifier.preprocess(self.all_data[self.field])#add back later if we need preprocessing
            # self.all_data["vectorised"]= vectors.toarray().tolist()
        else:
            self.all_data["preprocessed"] =self.all_data[self.field]
            self.all_data["backup_processed"] = self.classifier.preprocess(self.all_data[self.field])  # add back later if we need preprocessing
        self.classifier.set_model_field(self.field)
        #print()

    def retrain(self):
        #print("Retraining and updating")
        self.classifier.update_data(self.all_data["preprocessed"])#update the order of pre-proessed data
        self.classifier.train()

    def reorder(self):
        print("Reordering")
        #print(self.all_data["predictions"])
        if "LLM alone" in self.all_data.columns:
            try:
                self.all_data['predictions']=self.all_data['LLM alone'] + self.all_data['predictions']
            except:
                print("LLM inclusion failed")


        self.all_data.sort_values(by=['predictions'], ascending=False, inplace=True)
        #print(list(self.all_data["predictions"])[:20])


    def predict(self):
        #print("Predicting ")
        #print(self.all_data["predictions"].shape)

        self.all_data["predictions"]=self.classifier.predict(self.all_data)
        #print(self.all_data.index.values)


    def discover_labels(self):
        """
        Simulate sreening: Discover true gold-standard labels each time this function is called. Discover the labels for the references on top, which have hopefully been reordered to contain relavant references
        :return:
        """
        df=self.all_data[self.all_data["discovered_labels"] ==""]#not yet screened

        to_discover= list(df.index.values)[:self.time_to_retrain]#get the index values for the items to discover


        for i in to_discover:
            this_label=self.all_data.at[i, 'label']
            self.discovery_order.append(this_label)
            self.all_data.at[i, 'discovered_labels']=this_label


    def add_stats(self):
        df = self.all_data[self.all_data["discovered_labels"] == 1]#get positive labelled rows
        #print(df.shape)
        self.num_steps+= self.time_to_retrain#add how many references were screened at this point
        self.step_list.append(self.num_steps)#x axis: that's the variable we will use for plotting! Basically a list of points for x axis
        self.num_found_list.append(df.shape[0])# Y axis: that's the variable we will use for plotting! Basically a list of points for y axis

    def plot_stats(self,plottitle):
        df=pd.DataFrame(columns=["Screened References", "References found"])
        df["Screened References"]=self.step_list
        df["References found"] = self.num_found_list
        #print("REINDEX")
        #print(self.all_data.head())

        #print(self.discovery_order)


        # fig = px.line(df, x="Screened References", y="References found", title='Screening progress for {}'.format(plottitle),template='simple_white')
        #
        #
        # fig.show()
        return df, self.discovery_order



    def reorder_once(self):
        print("Reordering spreadsheet")
        self.time_to_retrain=len(list(self.all_data["label"]))#use all labels that we have

        self.discover_labels()  # "screen" 10 references. In similation, those refs are simply uncovered, ie added to a different column. If we have a screening UI and proper app, this method can be exchanged witha method that asks screeners to provide 10 labels
        self.retrain()  # technically not doing anything right now
        self.predict()  # calculate similarities and predict new order
        self.reorder()  # guess what that does LOL

        return self.all_data

    def simulate_learning(self, plottitle):
        print("Simulating active learning")
        while "" in list(self.all_data["discovered_labels"]):#while there are sill unlabelled references
            #print(".......Starting iteration...")
            self.discover_labels()#"screen" 10 references. In similation, those refs are simply uncovered, ie added to a different column. If we have a screening UI and proper app, this method can be exchanged witha method that asks screeners to provide 10 labels

            if self.stop_reordering == False:
                self.retrain()#technically not doing anything right now
                self.predict()#calculate similarities and predict new order
                self.reorder()#guess what that does LOL
            self.add_stats()#adding info to lists for plotting

            if  sum(self.all_data["label"])==len(list(self.all_data[self.all_data["discovered_labels"] == 1].index.values)) :#iff all positive labels were discovered
                self.stop_reordering =True
                print("Skipping learning - found recs")

            #print(self.all_data["discovered_labels"].value_counts())
        ranks, fullsteps= self.plot_stats(plottitle)


        return ranks, fullsteps