#This script can be sued to test how well SCANAR's relevancy ranking based on page rank and query matches performs in terms of finding relevant references early on in the screening process. It reads in a CSV file containing the SCANAR results, sorts the references by their relevancy score (appearances) and minimum page number, and then calculates the cumulative number of relevant references found as the screening progresses. Finally, it creates a line plot to visualize the screening progress.
#It assumes that the user has screened records in the order they are ranked by SCANAR and are present in the spreadsheet.
#This is not the AIDOC method. 

import pandas as pd
import plotly.express as px
df=pd.read_csv(r"ScanDatasets\NEWS_quantumscan.csv")
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

