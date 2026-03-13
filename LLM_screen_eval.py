

import os

from openai import OpenAI
import pandas as pd
import io
from sklearn.metrics import classification_report
import pandas as pd
import dotenv
import json

def my_eval(inpath):
  df=pd.read_csv(inpath)
  print(classification_report(df["label"], df["prediction"]))#or change colnames if needed

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    dotenv.load_dotenv()#you are expected to have a .env file with your OPENAI API key in the root directory/same folder as this script, and the .env file should be in your .gitignore to avoid sharing it. The .env file should contain a line like this with your OpenAI API key: API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    my_model = "gpt-5-mini"  # or gpt-4o, gpt-3.5-turbo, etc.
    outfolder = r"data\\LLM_predictions"

    managed_runs=json.load(open(r"managed_runs.json", "r"))#this is a json file where you can specify the input files you want to run the LLM evaluation on, the prompts and other variables
    #my_dataset="AI Technologies.csv"
    #my_dataset="GenAI Technologies 1 Trials"
    #my_dataset="MRC"
    done_data=["BPA"]


    for my_dataset in managed_runs.keys():
        if my_dataset not in done_data:
            pass
        else:
            print("Running LLM evaluation for dataset {}".format(my_dataset))
            
            filename = managed_runs[my_dataset]["path"]
            ti=managed_runs[my_dataset]["title_col"]
            ab=managed_runs[my_dataset]["abstract_col"]
            my_prompt = managed_runs[my_dataset]["prompt"]
            label_col=managed_runs[my_dataset]["label_col"]
            #################################################################Authentication
            print("Using API key to authenticate..")
            try:
                OPENAI_API_KEY = dotenv.get_key(dotenv.find_dotenv(), "API_KEY")
                client = OpenAI(
                    api_key=OPENAI_API_KEY
                )

            except Exception as e:
                # unknown error
                print("Error authenticating API key")
                raise e
            ###################################################Testing if it works
            # ask ChatGPT

            text = "How many fingers does a human hand have?"
            print("Testing API by asking: {}".format(text))

            completion = client.chat.completions.create(
                model=my_model,
                messages=[
                    {"role": "user", "content": '%s' % text}
                ]
            )

            openai_response = completion.choices[0].message.content
            print(openai_response)
            #
            ######upload a file


            try:
                df = pd.read_csv(filename, encoding='utf-8').fillna("")
                enc='utf-8'
            except:
                df = pd.read_csv(filename, encoding='windows-1252').fillna("")
                enc = 'windows-1252'

            ###############some params
            seed = 0
            ########################shuffel and split into dev and test
            #print(df["label"].value_counts())
            #print(df["decision"].value_counts())
            print(df.columns)
            print(df.shape)
            df = df.sample(frac=1, random_state=seed)  # shuffle and reindex
            df.reset_index(drop='True', inplace=True)

            backupdf=pd.read_csv("backup_BPA.csv", encoding=enc)
            ################ask GPT stuff about each row of data
            predictions = []
            justifications = []
            for i, row in df.iterrows():
                if i in backupdf.index:
                    predictions.append(backupdf.at[i, "prediction"])
                    justifications.append(backupdf.at[i, "Justification"])
                    print("Row {} already done, skipping..".format(i))
                    continue
                ####OLD######ti_abs_key="{} {} {}".format(row["Title"], row["Abstract"],row["Keywords"])
                ti_text=row.get(ti, "")
                ab_text=row.get(ab, "")
                ti_abs_key = "{} {}".format(ti_text, ab_text).strip()
                prompt = "{} {}".format(my_prompt, ti_abs_key)

                completion = client.chat.completions.create(
                    model=my_model,
                    messages=[
                        {"role": "user", "content": '%s' % prompt}
                    ]
                )
                openai_response = completion.choices[0].message.content
                if openai_response.lower().startswith("yes") or openai_response.lower().startswith("**YES**") or "YES" in openai_response.lower()[:10] or (len(ab_text)< 50 and ab != ""):#here an automated positive response is added if an abstract column was provided but if it's text is less than 50 characters long. If no abstract column is provided then classification is done as normal based on the title alone. This is because in some cases, such as with clinical trial registry entries, the abstract may be missing or very short, and we want to avoid false negatives in those cases.
                    predictions.append(1)
                else:
                    predictions.append(0)
                justifications.append(openai_response.replace("\n", " ").replace("  ", " "))
                print(str(i + 1) + ": " + openai_response)
                print("------------")
                if i % 100 == 0:
                    print("------BACKUP PREDICTIONS------")
                    ndf = pd.DataFrame()
                    ndf["prediction"] = predictions
                    ndf["Justification"] = justifications
                    ndf.to_csv("backup.csv", encoding=enc, index=False)

            print("FINAL PREDICTIONS")
            # [print(p) for p in predictions]

            df["LLM alone"] = predictions
            df["LLM Justification"] = justifications
            df.to_csv("data/backup.csv", index=False, encoding=enc)


            df.to_csv(os.path.join(outfolder,filename.split("\\")[-1]), index=False, encoding=enc)
            print(classification_report(df[label_col], df["LLM alone"]))

                #"You are a researcher screening news articles for inclusion in a literature analysis. The inclusion criteria are the following: Any article that describes a newly developed or upcoming health screening method or campaign for the early detection of diseases. Screening tests can be diagnostic; to detect cancer, dementia, HPV, or any other disease and health condition within a population. Screening methods may be offered or evaluated based on a whole population or people of selected age groups and locations. Any method, such as at-home, point of care, AI-supported, analysis of biomarkers and genes, or other methods are of interest, as long as they aim to detect diseases early. Answer YES if the article is relevant or unclear. Answer NO if it is not. Then reproduce the exact context from the paper that contained the information on which basis you made the decision. Here is the text of the article:"

                # prompt = "You are a researcher screening articles for inclusion for a systematic review. The inclusion criteria are the following: Any article that describes a method to automatically extract data, automatically label sentences, or does extractive summarisation of study characteristics of interest to systematic reviews in health. The included references describe data being extracted from clinical trials, epidemiologic studies, diagnostic accuracy studies, or other peer-reviewed literature related to evidence-based medicine. The automatically extracted characteristics can be population, intervention, outcomes or any other characteristic such as age, number of participants, and so on. Answer YES if the article is relevant or unclear. Answer NO if it is not. Then provide a short 2-sentence justification. Here is the title and abstract of the article: {}".format(
                #     ti_abs_key)
        
                # prompt = "You are a researcher screening news articles for inclusion in a literature analysis. The inclusion criteria are the following: Any article that describes a healthcare related technology using or quantum applications. This includes quantum computing and quantum sensing applications specifically within the healthcare sector. Technologies may also be using quantum mechanics such as superposition, entanglement, and interference. The aim of the technology should be to support healthcare or life sciences, including but not limited to drug discovery, health screening, diagnostic tools, disease detection, monitoring, assessment and prediction. Any technology utilising quantum principles, mechanics or computing with the intent of detecting or treating diseases is within the remit. Answer YES if the article is relevant or unclear. Answer NO if it is not. Then reproduce the exact context from the paper that contained the information on which basis you made the decision. Here is the text of the article: {}".format(
                #     ti_abs_key)
                # prompt = "You are a researcher screening references and clinical trial registry entries for inclusion in a literature analysis. The inclusion criteria are the following: Any reference that describes wearables used on humans, for prevention, diagnosis, prognosis, or treatment response of health conditions such as diseases, impairments, or disability. Devices worn by human subjects must be wearable, eg. biosensors worn on the body or attached to clothing, or in case of implants, worn invasively. Examples of wearables include, but are not limited to, smart watches, tattoos, lenses, glasses, earbuds, necklaces, smart patches, smart bracelets/bands, smart rings and wearable robots. Included reference should describe wearables being tested for health-relevant functions such as  remote monitoring of vital signs, support of rehabilitation, or chronic disease management. Exclude all references describing review articles, multiple conference proceedings, protocols, software development, materials development, VR and immersive technology, and other papers reporting no results on wearables on humans. Answer YES if the article is relevant to the inclusion criteria or unclear. Answer NO if it is not. Then reproduce the exact context from the paper that contained the information on which basis you made the decision. Here is the text of the article: {}".format(
                #     ti_abs_key)