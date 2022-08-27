# Import Modules
import pandas as pd
import importlib
import requests
import json
import csv

# Import API Secret
caseLaw = importlib.import_module("start_caselaw")
secretHeader = {'Authorization': 'Token %s' % caseLaw.secret}

df = pd.read_csv("data/SCDB_2021_01_caseCentered_Citation.csv",
                 encoding="iso-8859-1")
df['dateDecision'] = pd.to_datetime(df['dateDecision'])
filtered = df.loc[(df["dateDecision"] >= "01/01/2015") &
                  (df["dateDecision"] < "01/01/2020")]

# Get the cites for decisions
citations = filtered["sctCite"].tolist()

# Set up writing to a .csv
with open("data/opinionText.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=("sctCite", "text"))
    w.writeheader()

    for case in enumerate(citations):
        print("Pulling Case with citation %s" % case[1])
        case_opinion = {}
        case_opinion["sctCite"] = case[1]

        # Pull the case
        case = requests.get("https://api.case.law/v1/cases/?full_case=true&cite=%s&court=us&ordering=-decision_date&page_size=10" % case[1],
                            headers=secretHeader)

        # Convert case to readable format
        case = json.loads(case.content)

        # Pull text from all opinions for the case
        try:
            num_opinions = len(case["results"][0]
                               ["casebody"]["data"]["opinions"])
            case_opinion["text"] = ""
            for i in range(0, num_opinions):
                case_opinion["text"] += case["results"][0]["casebody"]["data"]["opinions"][i]["text"]

        # The error only shows up twice, and both of the citation searches have
        # no results. Will look into this later, but 2 out of 368 is not bad.
        except IndexError:
            print("ERROR on case with citation %s" % case_opinion["sctCite"])
            case_opinion["text"] = ""

        # Write the row
        w.writerow(case_opinion)
