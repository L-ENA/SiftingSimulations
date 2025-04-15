import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from buscarpy import generate_dataset
from buscarpy import calculate_h0
from buscarpy import retrospective_h0
import plotly.express as px
import plotly.graph_objects as go
import statistics

qrels=pd.read_csv(r"C:\Users\c1049033\PycharmProjects\ncl_medx\data\runs\Ensemble_Transgenerational_tiabs_Neural.csv")
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

    my_h0=retrospective_h0(seen_documents, qrels.shape[0], batch_size=500, recall_target=0.95)
    my_h01 = retrospective_h0(seen_documents, qrels.shape[0], batch_size=500, recall_target=0.95, bias=1.5)
    my_h02=retrospective_h0(seen_documents, qrels.shape[0], batch_size=500, recall_target=0.95, bias=2)

    work_saved.append(1-(my_h0['batch_sizes'][-1]/qrels.shape[0]))
    discoveredAtstop = sum(seen_documents[:my_h0['batch_sizes'][-1]])
    true_recall.append(discoveredAtstop / total_incl)
    if true_recall[-1]<0.95:
        print("Lower recall bias {} @ {}".format("0", true_recall[-1]))

    work_saved2.append(1 - (my_h02['batch_sizes'][-1] / qrels.shape[0]))
    discoveredAtstop2 = sum(seen_documents[:my_h02['batch_sizes'][-1]])
    true_recall2.append(discoveredAtstop2 / total_incl)
    if true_recall2[-1]<0.95:
        print("Lower recall bias {} @ {}".format("2", true_recall2[-1]))

    work_saved1.append(1 - (my_h01['batch_sizes'][-1] / qrels.shape[0]))
    discoveredAtstop1 = sum(seen_documents[:my_h01['batch_sizes'][-1]])
    true_recall1.append(discoveredAtstop1 / total_incl)
    if true_recall1[-1]<0.95:
        print("Lower recall bias {} @ {}".format("1.5", true_recall1[-1]))


fig = px.scatter(x=work_saved, y=true_recall, title="Early Stopping: Work Saved vs. True recall", labels={'x': 'Percentage of data not needed to be seen', 'y':'True underlying recall'}, template='simple_white')
fig.data[-1].name = 'Buscar@95%'
fig.add_scatter(x=[statistics.mean(work_saved)],
                y=[statistics.mean(true_recall)],
                marker=dict(
                    color='blue',
                    size=15
                ),
               name='Mean Buscar@95')

df=pd.DataFrame()
df["work_saved1"]=work_saved1
df["true_recall1"]=true_recall1
df.sort_values("work_saved1", ascending=True, inplace=True)

fig.add_scatter(x=df["work_saved1"], y=df["true_recall1"],mode='markers',

               name='Buscar@95 Bias 1.5')
fig.add_scatter(x=[statistics.mean(work_saved1)],
                y=[statistics.mean(true_recall1)],
                marker=dict(
                    color='green',
                    size=15
                ),
               name='Mean Buscar@95 Bias 1.5')

df=pd.DataFrame()
df["work_saved2"]=work_saved2
df["true_recall2"]=true_recall2
df.sort_values("work_saved2", ascending=True, inplace=True)
fig.add_scatter(x=df["work_saved2"], y=df["true_recall2"],mode='markers',

               name='Buscar@95 Bias 2')
fig.add_scatter(x=[statistics.mean(work_saved2)],
                y=[statistics.mean(true_recall2)],
                marker=dict(
                    color='purple',
                    size=15
                ),
               name='Mean Buscar@95 Bias 2')

fig.update_traces(showlegend = True)
fig.update_traces(marker_line_color="black")

fig.show()