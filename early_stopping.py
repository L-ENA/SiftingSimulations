import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from buscarpy import generate_dataset
from buscarpy import calculate_h0
from buscarpy import retrospective_h0
import plotly.express as px
import plotly.graph_objects as go
import statistics
import os
gaincurve_dir=r"data\\gaincurves\\buscar\\"
qrels_dir=r"data\\runs\\"
results_summary=pd.DataFrame(columns=["dataset","run", "work_saved_bias_1", "true_recall_bias_1", "work_saved_bias_1.5", "true_recall_bias_1.5", "work_saved_bias_2", "true_recall_bias_2"])
plot_separate_results=True#if true, a  plot is created showinf single dataset results. Note, this script create a summary df and plot_all_runs.py plots them together, which is more informative and less cluttered than individual plots, but this can be useful for sanity checks and to understand the variability across runs and datasets.

dataset_runs=['LLM_1.0_GenAI Technologies 2 UKRI_tiabs_Neural.csv', 'LLM_1.0_GenAI Technologies 3 NIHR_tiabs_Neural.csv', 'LLM_1.0_Woman Health UKRI_tiabs_Neural.csv', 'LLM_AI Technologies_ScientificTitle_Neural.csv',   'LLM_GenAI Technologies 1 Trials_tiabs_Neural.csv', 'LLM_new_MRC_tiabs_Neural.csv', 'LLM_MRP_tiabs_Neural.csv', 'LLM_Rapid Biomed Genetic Engineering_tiabs_Neural.csv', 'LLM_Rapid Biomed Tissues Devices_tiabs_Neural.csv', 'LLM_Rapid Biomed TX delivery_tiabs_Neural.csv']
#dataset_runs=['LLM_BPA_tiabs_Neural.csv', 'LLM_Fluoride_tiabs_Neural.csv', 'LLM_PFOA_tiabs_Neural.csv', 'LLM_Transgenerational_tiabs_Neural.csv']
#data_name="LLM_GenAI Technologies 1 Trials_tiabs_Neural"
for data_name in dataset_runs:
    qrels=pd.read_csv(r"{}{}".format(qrels_dir, data_name))
    work_saved=[]
    true_recall = []

    work_saved1=[]
    true_recall1 = []

    work_saved2=[]
    true_recall2 = []

    for col in qrels.columns:

        print(col)
        seen_documents=qrels[col]
        total_incl = sum(seen_documents)

        my_h0=retrospective_h0(seen_documents, qrels.shape[0], batch_size=20, recall_target=0.95, plot=False)
        my_h01 = retrospective_h0(seen_documents, qrels.shape[0], batch_size=20, recall_target=0.95, bias=1.5, plot=False)
        my_h02=retrospective_h0(seen_documents, qrels.shape[0], batch_size=20, recall_target=0.95, bias=2, plot=False)

        work_saved_bias_1 = 1-(my_h0['batch_sizes'][-1]/qrels.shape[0])
        work_saved.append(work_saved_bias_1)
        discoveredAtstop = sum(seen_documents[:my_h0['batch_sizes'][-1]])
        true_recall_bias_1 = discoveredAtstop / total_incl
        true_recall.append(true_recall_bias_1)
        if true_recall_bias_1<0.95:
            print("Lower recall bias {} @ {}".format("0", true_recall_bias_1))

        work_saved_bias_2 = 1 - (my_h02['batch_sizes'][-1] / qrels.shape[0])
        work_saved2.append(work_saved_bias_2)
        discoveredAtstop2 = sum(seen_documents[:my_h02['batch_sizes'][-1]])
        true_recall_bias_2 = discoveredAtstop2 / total_incl
        true_recall2.append(true_recall_bias_2)
        if true_recall_bias_2<0.95:
            print("Lower recall bias {} @ {}".format("2", true_recall_bias_2))

        work_saved_bias_1_5 = 1 - (my_h01['batch_sizes'][-1] / qrels.shape[0])
        work_saved1.append(work_saved_bias_1_5)
        discoveredAtstop1 = sum(seen_documents[:my_h01['batch_sizes'][-1]])
        true_recall_bias_1_5 = discoveredAtstop1 / total_incl
        true_recall1.append(true_recall_bias_1_5)
        if true_recall_bias_1_5<0.95:
            print("Lower recall bias {} @ {}".format("1.5", true_recall_bias_1_5))

        results_summary.loc[len(results_summary)] = {
            "dataset": data_name,
            "run": col,
            "work_saved_bias_1": work_saved_bias_1,
            "true_recall_bias_1": true_recall_bias_1,
            "work_saved_bias_1.5": work_saved_bias_1_5,
            "true_recall_bias_1.5": true_recall_bias_1_5,
            "work_saved_bias_2": work_saved_bias_2,
            "true_recall_bias_2": true_recall_bias_2,
        }

    if plot_separate_results:
        fig = px.scatter(x=work_saved, y=true_recall, title="Early Stopping at 95% estimated recall: Records not needed to be screened vs. True underlying recall", labels={'x': 'Percentage of data not needed to be seen', 'y':'True underlying recall'}, template='simple_white')
        fig.data[-1].name = 'CMH at Bias 1'
        fig.add_scatter(x=[statistics.mean(work_saved)],
                        y=[statistics.mean(true_recall)],
                        marker=dict(
                            color='blue',
                            size=15
                        ),
                    name='Mean CMH at Bias 1')

        df=pd.DataFrame()
        df["work_saved1"]=work_saved1
        df["true_recall1"]=true_recall1
        df.sort_values("work_saved1", ascending=True, inplace=True)

        fig.add_scatter(x=df["work_saved1"], y=df["true_recall1"],mode='markers',

                    name='CMH at Bias 1.5')
        fig.add_scatter(x=[statistics.mean(work_saved1)],
                        y=[statistics.mean(true_recall1)],
                        marker=dict(
                            color='green',
                            size=15
                        ),
                    name='Mean CMH at Bias 1.5')

        df=pd.DataFrame()
        df["work_saved2"]=work_saved2
        df["true_recall2"]=true_recall2
        df.sort_values("work_saved2", ascending=True, inplace=True)
        fig.add_scatter(x=df["work_saved2"], y=df["true_recall2"],mode='markers',

                    name='CMH at Bias 2')
        fig.add_scatter(x=[statistics.mean(work_saved2)],
                        y=[statistics.mean(true_recall2)],
                        marker=dict(
                            color='purple',
                            size=15
                        ),
                    name='Mean CMH at Bias 2')

        # Horizontal reference line at true recall = 0.95
        fig.add_scatter(x=[0, 1],
                        y=[0.95, 0.95],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='Recall 0.95')

        fig.update_traces(showlegend = True)
        fig.update_traces(marker_line_color="black")

        fig.show()
        fig.write_html(os.path.join(gaincurve_dir, "{}.html".format(data_name)))
        plt.close()

results_summary.to_csv(os.path.join(gaincurve_dir, "results_summary_hs_data.csv"), index=False)
