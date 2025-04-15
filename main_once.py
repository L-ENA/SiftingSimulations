import pandas as pd
from ActiveLearner import ActiveLearner
from neural_classifier import SPECTER_CLS
from OLD_ML_classifier import ML_CLS
import nltk
nltk.download('punkt_tab')
#import plotly.express as px
import random
from eval_measures import do_eval
import os

mypath=r"C:\Users\c1049033\PycharmProjects\ncl_medx\data\LLM predictions\26_03_2025_NSC_data.csv"
goldpath=r"C:\Users\c1049033\PycharmProjects\ncl_medx\data\datasets\Feb Extended Health Screening.csv"
temppath=r"H:\Downloads\12_03_2025_NSC_data_temp.csv"

myfield="tiabs"
mymodel="Neural"
nruns=5
data_name="Ext_NSC_{}_{}".format(myfield, mymodel)

df=pd.read_csv(mypath, encoding='utf-8-sig').fillna("")
df["tiabs"]=df["title"]+ " " + df["fulltext"]
#df["tiabs"]=[a[:2000].replace("\n", " ").replace("  ", " ") for a in list(df["tiabs"])]
df["tiabs"]=[a[:2000].replace("\n", " ").replace("  ", " ") for a in list(df["tiabs"])]
#df["tiabs"]=[re.sub(r'(<jats:.+?>)|(</jats:.+?>)', r" ", a) for a in df['tiabs']]
df["tiabs"]=[a.replace("\n", " ").replace("  ", " ") for a in list(df["tiabs"])]
df.to_csv(mypath, index=False)

#################when predicting on unscreened or partly screened data:
label_df=pd.read_csv(goldpath).fillna("")
label_df["tiabs"]=label_df["title"]+ " " + label_df["fulltext"]
# tiabs=label_df["tiabs"]
# decs=label_df["decision"]
label_df["ID"]=["{}_{}".format("L", i) for i in label_df["ID"]]

data = pd.concat([df, label_df])
#data = pd.read_csv(mypath).fillna("")#alternatively read from csv

data['label']=data["decision"]
    #data["tiabs"] = data["title"] + " " + data["abstract"]
#print(data.loc[0]["tiabs"])

for s in range(nruns):


    data = data.sample(frac=1, random_state=s)
    incls = data.index[data['label'] == 1].tolist()
    random.seed(s)
    starters = random.sample(incls, len(incls))
    print(starters)
    sort_df = [0 if i not in starters else 1 for i in data.index]
    data['temp'] = sort_df
    data.sort_values("temp", ascending=False, inplace=True)
    data.reset_index(drop='True', inplace=True)
    data = data.drop('temp', axis=1)
    if s % 5 != 0:
        classifier = SPECTER_CLS
    else:
        classifier= ML_CLS

    al = ActiveLearner(classifier, data, field=myfield, model_name=mymodel, do_preprocess=False)
    output = al.reorder_once()
    output[str(s)]=output["predictions"]
    output.to_csv(temppath, index=False)

df=pd.read_csv(temppath)
df=df[df["reviewed"]!= "yes"]
to_keep=["title","description","media","decision","resolved_link","link","date","fulltext","min_page","appearances","queries", "OLD_predictions",'New_predictions', "LLM alone", "LLM Justification"]
df['OLD_predictions'] = df[[str(i) for i in range(nruns)]].mean(axis=1)
keys=[str(i) for i in range(nruns)]
keys.append("LLM alone")

df['New_predictions'] = df[keys].mean(axis=1)
df = df[to_keep]
print(df["OLD_predictions"])
df.sort_values("New_predictions", ascending=False, inplace=True)
df.to_csv(mypath, encoding="utf-8-sig")
