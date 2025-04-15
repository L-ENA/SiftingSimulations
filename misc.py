import pandas as pd

df=pd.read_csv(r"H:\Downloads\ReferenceExport (16).csv")
titles=set()
todel=[]
print(df.shape)
for i,row in df.iterrows():
    tt=row["Citation"]
    if tt in titles:
        todel.append(row["ActiveScreener Id"])
    else:
        titles.add(tt)
print(len(todel))
outdf=pd.DataFrame()
outdf["ActiveScreener Id"]=todel
outdf.to_csv(r"C:\Users\c1049033\Documents\CE\sunproduct_duplicates.csv", index=False)