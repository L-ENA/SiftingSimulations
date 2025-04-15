import pandas as pd
import plotly.express as px
df=pd.read_csv(r"C:\Users\c1049033\Documents\ScanDatasets\NEWS_quantumscan.csv")
print(df.shape)
df=df.sort_values(by=['appearances', 'min_page'],ascending=[False, True])
decisions=list(df["label"])
sums=[]
count=0
for i,val in enumerate(decisions):
    if val != 'Exclude':
        count +=1
    sums.append(count)
newdf=pd.DataFrame()
newdf["References found"]=sums
newdf["Screened References"]=newdf.index
fig = px.line(newdf, x="Screened References", y="References found", title='Screening progress for {}'.format("SCANAR max apprearances relevancy"),template='simple_white')
print(sums)
fig.show()

