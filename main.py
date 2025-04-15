import pandas as pd
from ActiveLearner import ActiveLearner
from neural_classifier import SPECTER_CLS
import plotly.express as px
import random
from eval_measures import do_eval
import os
from classifiers_backup import MLClassifiers

if __name__ == '__main__':

    random_states=range(15)
    
    data = pd.read_csv(r"C:\Users\c1049033\PycharmProjects\ncl_medx\data\datasets\full_mined_labelled.csv").fillna("")
    data['label']=data["label"]#change if the input spreadsheet has data elsewhere. There always needs to be a 'label' column with binary labels

    # data["tiabs"] = data["title"] + " " + data["fulltext"][:2000]#merge title abstract if needed and set the myfield variable to point to it
    # myfield="tiabs"


    myfield="ScientificTitle"#the text on which we run simulation

    mymodel="Neural"

    data_name="Demo_{}_{}".format(myfield, mymodel)





    ##################################################NEURAL MODEL
    classifier = SPECTER_CLS
    #classifier=MLClassifiers

    summary_df=pd.DataFrame()
    steps_df = pd.DataFrame()
    for seed in random_states:
        data = data.sample(frac=1, random_state=seed)
        incls=data.index[data['label'] == 1].tolist()
        random.seed(seed)
        starters=random.sample(incls, 5)#n starting seeds
        print(starters)
        sort_df=[0 if i not in starters else 1 for i in data.index]
        data['temp']=sort_df
        data.sort_values("temp", ascending=False, inplace=True)
        data.reset_index(drop='True', inplace=True)
        data=data.drop('temp', axis=1)
        print(data["label"][:2])


        al = ActiveLearner(classifier, data, field=myfield, model_name=mymodel, do_preprocess=False)
        #al = ActiveLearner(classifier, data, field=myfield, model_name=mymodel, do_preprocess=True)

        my_df, fullsteps=al.simulate_learning(plottitle="AI simulation")#ecample for simulation, can still be used to provide fancy plot to the user to see how the model would have reacted tto their data in active learning scenario
        if summary_df.shape[0]==0:
            summary_df["Screened References"]=my_df["Screened References"]

        summary_df["Results_{}".format(seed)]=my_df["References found"]
        steps_df["Results_{}".format(seed)] = fullsteps

    fig = px.line(summary_df, x='Screened References', y=summary_df.columns[-len(random_states):],template='simple_white')
    fig.update_layout(legend_title_text='Runs')
    fig.show()
    summary_df.to_csv("data//stats_nruns_{}.csv".format(data_name), index=False)
    outp="data//runs//{}.csv".format(data_name)
    steps_df.to_csv(outp, index=False)

    
    results = do_eval(outp)
    resp=os.path.join(r"C:\Users\c1049033\PycharmProjects\ncl_medx\data\global", "{}.csv".format(data_name))

    results.to_csv(resp, index=False)



    # px.line(df, x="Screened References", y="References found", title='Screening progress for {}'.format(plottitle),
    #         template='simple_white')
    # output= al.reorder_once()
    # output.to_csv("data//reordered_once.csv", index=False)

    ####################################################Can safely ignore this. These lines look where data was missing from the scan and supplement it with the mined data. I guess at runtime with new projects we won;t have that yet.
    # ints=data["Interventions"]
    # mined=data["mined_intervention_control"]
    # new=[ d if d != "" else mined[i] for i, d in enumerate(ints) ]#use mined data if no intervation pulled from scan. Totally optional
    # data["Interventions"]=new
    #########################################################

    ####################################Filter model
    # classifier = regexClassifier
    # al = ActiveLearner(classifier, data, field="ScientificTitle", model_name="Filter")
    # #al = ActiveLearner(classifier, data, field="Interventions", model_name="Filter")
    # al.simulate_learning()

    ################################################Random as reference
    # classifier = emptyClassifier
    # al = ActiveLearner(classifier, data, field="ScientificTitle", model_name="Random", do_preprocess=False)
    # al.simulate_learning()


def remit_new(simulate=False):
    mypath = "data//UK_Calls_new.csv"
    remits = ["Advanced Therapies"	,"Biomedical engineering"	,"Drug Therapy & medical device combination",	"Diagnostics",	"Artificial Intelligence",	"Gut health-microbiome-nutrition"]
    for r in remits:
        if r != "":
            print(r)
            data = pd.read_csv(mypath).fillna("")
            data = data.sample(frac=1, random_state=48)
            data.reset_index(drop=True, inplace=True)

            data["label"]=0
            data.loc[data[r] == "Y", 'label'] = 1
            if not simulate:
                print("reordering")
                print(data.shape)
                print(sum(data["label"]))

                classifier = SPECTER_CLS
                al = ActiveLearner(classifier, data, field="Summary", model_name="Neural", do_preprocess=True)
                output = al.reorder_once()
                output.rename(columns={"predictions": "{}_predictions_Summary".format(r.replace(" ", "_"))}, inplace=True)
                output.to_csv(mypath, index=False)
            else:
                print("simulating")

                data = data.drop(data[data.Include == ""].index)
                print(data.shape)
                print(sum(data["label"]))
                classifier = SPECTER_CLS
                al = ActiveLearner(classifier, data, field="Description", model_name="Neural", do_preprocess=False)
                al.simulate_learning(r)

def precompute(sents):
    from sentence_transformers import util, SentenceTransformer
    model_name = 'sentence-transformers/allenai-specter'
    model = SentenceTransformer(model_name)

    emb_source = model.encode(list(sents))  # get our data column
    cos_sim = util.pytorch_cos_sim(emb_source, emb_source)  # .diagonal().tolist()#all similarities
    return cos_sim

def sort_by_remit(simulate=False):
    mypath="H://Downloads//sonia_new.csv"
    mycol="Category"
    lbl="label"
    refcol="Tiab"
    data = pd.read_csv(mypath).fillna("")
    txts=data[refcol]
    print("precomputing embeddings")
    #cosims=precompute(txts)


    remits=data[mycol].unique()
    print(remits)
    # abstracts=[t[:900] for t in data["Description"]]
    # data["abbrev"]=abstracts
    # data.to_csv(mypath, index=False)

    for r in remits:
        if r != "":
            print(r)
            data = pd.read_csv(mypath).fillna("")
            data = data.sample(frac=1, random_state=48)
            data.reset_index(drop=True, inplace=True)

            data[lbl]=0
            data.loc[data[mycol] == r, lbl] = 1
            if not simulate:
                print("reordering")
                print(data.shape)
                print(sum(data[lbl]))

                classifier = SPECTER_CLS
                #al = ActiveLearner(classifier, data, field=refcol, model_name="Neural", do_preprocess=False, precomputed=cosims)
                al = ActiveLearner(classifier, data, field=refcol, model_name="Neural", do_preprocess=False)
                output = al.reorder_once()
                output.rename(columns={"predictions": "{}_predictions_Tiab2".format(r.replace(" ", "_"))}, inplace=True)
                output.to_csv(mypath, index=False)
            else:
                print("simulating")

                data = data.drop(data[data[mycol] == ""].index)
                print(data.shape)
                print(sum(data[lbl]))
                classifier = SPECTER_CLS
                al = ActiveLearner(classifier, data, field="Title", model_name="Neural", do_preprocess=True)
                al.simulate_learning(r)


#sort_by_remit(simulate=False)
#remit_new(simulate=True)
#add_embedding("H://Downloads//fixed.csv", ["Abstract",	"Title"])



