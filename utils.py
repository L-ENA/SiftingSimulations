import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import util


def calculate_similarity_single_sent(pos_idx, cos_sim):
    """
    pos_idx: list with lindex calues of positively-labelled rows
    ##target_list: list with strings to claculate similarities on

    returns: list of length target_list, with float values between 0 and 1 corresponding to cosine similarity (1=similar).

    Example:
    idx_pos=[0,2,4]
    inputs=["Artificial Intelligence With DEep Learning on COROnary Microvascular Disease",
            "Neuronal Mechanisms of Human Episodic Memory",
            "Polyp REcognition Assisted by a Device Interactive Characterization Tool - The PREDICT Study",
            "Artificial Intelligence With DEep Learning on COROnary Microvascular Disease",
            "Artificial intelligence and machine learning for Covid 19",
            "Effects of a Mindfulness-Based Eating Awareness Training online intervention in adults of obesity: A randomized controlled trials",
            "Prediction of Phakic Intraocular Lens Vault Using Machine Learning",
            "AI system for egg OFC Prediction System of Infants"]

    calculate_similarity_single_sent(idx_pos, inputs,model)

    """
    # emb_source= my_model.encode(target_list)#get our data column

    # print(emb_source.shape)
    # print("Calculating similarity matrix...")

    # print(cos_sim.shape)

    all_sims = []

    for i in tqdm(cos_sim):  # for each input and its pairwise similarities
        avg_for_record = []  # list to store all pairwwise similarities of this field with the positively labelled fields
        for ind in pos_idx:  # for each positive labelled record
            avg_for_record.append(i[ind].item())  # add the cosine similarity
            # print(i[ind])
        try:
            all_sims.append(sum(avg_for_record) / len(avg_for_record))  # average similarity is used for now, but could use median etc
        except:
            all_sims.append(0)
        # print(sum(avg_for_record)/len(avg_for_record))
        # print("-----")
    # print(all_sims)
    return all_sims

def add_embedding(path_to_df, refcols):
    df=pd.read_csv(path_to_df).fillna("")
    print(df.shape)
    from sentence_transformers import util, SentenceTransformer
    model_name = 'sentence-transformers/allenai-specter'
    model = SentenceTransformer(model_name)
    for c in tqdm(refcols):
        ens = model.encode(list(df[c]))
        llst=[list(e) for e in ens]

        df["Emb_{}".format(c)]=llst
    print(df.columns)
    df.to_pickle(path_to_df.replace(".csv",".aidoc"))

def clean_data():
    df=pd.read_csv(r"C:\Users\c1049033\Documents\ScanDatasets\rapid_biomed_final_all.csv")
    print(df.shape)
    rows=[]
    titles=set()
    for i,row in df.iterrows():
        if row["Title"] not in titles:
            titles.add(row["Title"])
            rows.append(i)

    df2 = df[df.index.isin(rows)]
    # print(list(df2.index)[:200])
    print(df2.shape)
    df2.to_csv(r"C:\Users\c1049033\Documents\ScanDatasets\rapid_biomed_final_all_deduped.csv", index=False)
#clean_data()