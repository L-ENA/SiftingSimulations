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
    
    # data = pd.read_csv(r"data\LLM_predictions\AI Technologies.csv").fillna("")
    # data['label']=data["label"]#change if the input spreadsheet has data elsewhere. There always needs to be a 'label' column with binary labels
    # dsname="AI Technologies"
    # myfield="ScientificTitle"#the text on which we run simulation
    # mymodel="Neural"
    # data_name="LLM_{}_{}_{}".format(dsname,myfield, mymodel)

    # data = pd.read_csv(r"data\LLM_predictions\GenAI Technologies 1 Trials.csv").fillna("")
    # data['label']=data["label"]#change if the input spreadsheet has data elsewhere. There always needs to be a 'label' column with binary labels
    # dsname="GenAI Technologies 1 Trials"
    # myfield="tiabs"#the text on which we run simulation
    # mymodel="Neural"
    # data_name="LLM_{}_{}_{}".format(dsname,myfield, mymodel)

    data = pd.read_csv(r"data\LLM_predictions\mrc_labelled.csv").fillna("")
    data['label']=data["label"]#change if the input spreadsheet has data elsewhere. There always needs to be a 'label' column with binary labels
    dsname="MRC"
    myfield="tiabs"#the text on which we run simulation
    mymodel="Neural"
    data_name="LLM_{}_{}_{}".format(dsname,myfield, mymodel)

    
    # data["tiabs"] = data["title"] + " " + data["fulltext"][:2000]#merge title abstract if needed and set the myfield variable to point to it
    # myfield="tiabs"





    ##################################################NEURAL MODEL
    classifier = SPECTER_CLS
    #classifier=MLClassifiers

    summary_df=pd.DataFrame()
    steps_df = pd.DataFrame()
    for seed in random_states:
        print("Running simulation with random seed {}".format(seed))
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
    fig.update_layout(legend_title_text='Runs', title=data_name)
    fig.show()
    
    # Save gain curve plot
    gaincurve_dir = "data//gaincurves//LLM"
    os.makedirs(gaincurve_dir, exist_ok=True)
    fig.write_html(os.path.join(gaincurve_dir, "{}.html".format(data_name)))
    
    summary_df.to_csv("data//stats_nruns_{}.csv".format(data_name), index=False)
    outp="data//runs//{}.csv".format(data_name)
    steps_df.to_csv(outp, index=False)

    
    results = do_eval(outp)
    resp=os.path.join(r"data//global", "{}.csv".format(data_name))

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




#sort_by_remit(simulate=False)
#remit_new(simulate=True)
#add_embedding("H://Downloads//fixed.csv", ["Abstract",	"Title"])



