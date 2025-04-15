import pandas as pd
from tqdm import tqdm

def describe_df(my_df,name):
    print('Analysing {}:\n'.format(name))
    for c in list(my_df.columns):
        print("Data <{}> Column <{}> has {} entries. {} are empty, {} have values. There are {} unique values in the column.".format(
                name,
                c,
                my_df.shape[0],
                my_df[c].isna().sum(),
                my_df.shape[0] - my_df[c].isna().sum(),
                len(my_df[c].unique())))

        print(my_df[c].value_counts()[:3])
        print("-----------\n")

def pull_data(this_id, this_column, dfs):
    """
    Get a specific column's value for a specific trial id
    :param this_id:
    :param this_column:
    :param this_df:
    :return: value
    """

    #try:
    for this_df in dfs:
        if this_id in list(this_df['Trial Identifier']) and this_column in list(this_df.columns):
            idx_at_id = list(this_df[this_df['Trial Identifier'] == this_id].index.values)[0]

            return this_df.loc[idx_at_id, this_column]#if this trial id is in this df and if this df actually contains the column we are searching for, then we can pull the value

    #print("Value for {} in {} not found".format(this_id, this_column))
    # except:
    #
    #     print("Error, value for {} in {} not found".format(this_id,this_column))
    #     pass
    return ""

def read_data_aiscan(path_to_file,out_path):

    ###############read dataset
    includes=pd.read_excel(path_to_file,sheet_name="To add")
    all_1 = pd.read_excel(path_to_file,sheet_name="AI Scan Jan20 to Jun20")  # Specific Condition Name;Developer Name;Trial name, dates, ID and URL
    print(list(all_1.columns))
    #describe_df(all_1, "AI Scan Jan20 to Jun20")

    all_2=pd.read_excel(path_to_file,sheet_name="AI Jul20 to Dec20")#Specific Condition;Developer Name;Country of Development; Trial name, dates, ID and URL
    print(list(all_2.columns))
    #describe_df(all_2, "AI Jul20 to Dec20")

    print(list(includes.columns))
    ###############clean dataset
    dfs=[includes,all_1,all_2]#all data choices
    cols=['Developer Name', 'Developer Profile', 'Product Name', 'Source link to product', 'Product Description', 'Type of Scanning/ Medical Imaging (if applicable)', 'Clinical Condition Area', 'Specific Condition Name', 'Classification of Technology', 'Country of Development', 'Development stage', 'Overall Regulatory Approval', 'Trial Start Date', 'Trial End Date', 'Trial Name', 'TrialURL']
    gold_data=pd.DataFrame(columns=['Developer Name', 'Developer Profile', 'Product Name', 'Source link to product', 'Product Description', 'Type of Scanning/ Medical Imaging (if applicable)', 'Clinical Condition Area', 'Specific Condition Name', 'Classification of Technology', 'Country of Development', 'Development stage', 'Overall Regulatory Approval', 'Trial Start Date', 'Trial End Date', 'Trial Name', 'TrialURL'])

    ids=set()#get all ids (unique)
    ids.update(list(all_1["Trial Identifier"]))
    ids.update(list(all_2["Trial Identifier"]))
    ids.update(list(includes["Trial Identifier"]))
    all_ids=list(ids)
    all_labels=[1 if i in list(includes["Trial Identifier"]) else 0 for i in all_ids]#assign include labels

    gold_data["Trial Identifier"]=all_ids#the gold-standard data set df
    gold_data["label"]=all_labels

    for i,row in tqdm(gold_data.iterrows()):
        for c in cols:
            gold_data.loc[gold_data.index[i], c]= pull_data(row["Trial Identifier"], c, dfs)#pull info from  all 3 speadsheets together

    gold_data.to_csv(out_path,index=False)

    print(gold_data.shape)


##########mergin the old project-only data#################
# gold_standard_all=r'C:\Users\lena.schmidt\PycharmProjects\ncl_medx\data\data_raw.xlsx'
# out_path=r'C:\Users\lena.schmidt\PycharmProjects\ncl_medx\data\data.csv'
# read_data_aiscan(gold_standard_all,out_path)

################Map labels from labelled to new data (whole-registry data)
in_data=r"C:\Users\lena.schmidt\PycharmProjects\ncl_medx\data\data_mined.csv"#project data gold-standard with labels
scan_data=r"C:\Users\lena.schmidt\PycharmProjects\ncl_medx\data\complete_all.csv"#data with full fields
out_path=r'C:\Users\lena.schmidt\PycharmProjects\ncl_medx\data\full_mined_labelled.csv'
def map_gold_labels(inpath,datapath,outpath):
    gold=pd.read_csv(inpath)
    print("Gold has {} entries".format(gold.shape[0]))

    scan=pd.read_csv(datapath)
    print("Scan has {} entries".format(scan.shape[0]))

    scan["label"]=""#empty field

    label_dict={row["Trial Identifier"]:row["label"] for i,row in gold.iterrows()}
    print("Found {} gold-labels and IDs to map".format(len(label_dict.keys())))

    for i,row in scan.iterrows():
        this_label=label_dict.get(row["MainID"],"")
        #print(this_label)
        if this_label!= "":
            scan.at[i,"label"]=this_label
        else:
            scan.at[i, "label"] = "UNK"
            print("ID <{}> has no label, filling <UNK> at index {}".format(row["MainID"],i))
    scan.to_csv(outpath, index=False)

map_gold_labels(in_data,scan_data,out_path)