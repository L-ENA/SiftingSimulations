#CODE SOURCE:
# https://github.com/WojciechKusa/normalised-precision-at-recall/blob/main/notebooks/evaluation.py



import numpy as np
import pandas as pd
import statistics


def average_precision_for_query(
    rankings: dict[str, float], qrels: dict[str, int]
) -> float:
    """Calculates the average precision for a given query.
    :param rankings: a dictionary of document ids and their corresponding scores
    :param qrels: a dictionary of document ids and their corresponding relevance labels
    :return: mean average precision for a given query
    """
    average_precision = 0
    tp, fp = 0, 0

    relevant_docs = sum(qrels.values())

    for rank, (doc_id, _) in enumerate(rankings.items()):
        if qrels.get(doc_id, 0) == 1:  # Document is relevant
            tp += 1
            precision = tp / (tp + fp)
            average_precision += precision
        else:
            fp += 1

    return average_precision / relevant_docs if relevant_docs > 0 else 0


def sqrt_n_precision_at_recall_for_query(
    rankings: dict[str, float], qrels: dict[str, int], recall_level: float = 0.95
) -> float:
    n_precision = n_precision_at_recall_for_query(rankings, qrels, recall_level)

    return np.sqrt(n_precision)


def n_precision_at_recall_for_query(
    rankings: dict[str, float], qrels: dict[str, int], recall_level: float = 0.95
) -> float:
    n_precision = None
    tp, fp, tn, fn = 0, 0, 0, 0

    relevant_docs = sum(qrels.values())
    total_docs = len(qrels)
    non_relevant_docs = total_docs - relevant_docs

    for rank, (doc_id, _) in enumerate(rankings.items()):
        if qrels.get(doc_id, 0) == 1:  # Document is relevant
            tp += 1
        else:
            fp += 1

        recall = tp / (tp + fn + relevant_docs - tp)

        if recall >= recall_level:
            tn = non_relevant_docs - fp
            n_precision = (tp * tn) / ((fp + tn) * (tp + fp))
            break

    return n_precision


def precision_at_recall_for_query(
    rankings: dict[str, float], qrels: dict[str, int], recall_level: float = 0.95
) -> float:
    precision = 0
    tp, fp, tn, fn = 0, 0, 0, 0

    relevant_docs = sum(qrels.values())

    for rank, (doc_id, _) in enumerate(rankings.items()):
        if qrels.get(doc_id, 0) == 1:  # Document is relevant
            tp += 1
        else:
            fp += 1

        recall = tp / (tp + fn + relevant_docs - tp)

        if recall >= recall_level:
            precision = tp / (tp + fp)
            break

    return precision

def scores_at_percentage(rankings: dict[str, float], qrels: dict[str, int], completeness: float = 0.50):
    tp, fp, tn, fn = 0, 0, 0, 0
    relevant_docs = sum(qrels.values())
    total_docs = len(qrels)
    non_relevant_docs = total_docs - relevant_docs
    precision=0
    recall=0

    max_steps=int(total_docs*completeness)-1#-1 sonce we later enumerate and enumerations start with 0

    for rank, (doc_id, _) in enumerate(rankings.items()):
        if qrels.get(doc_id, 0) == 1:  # Document is relevant
            tp += 1
        else:
            fp += 1

        recall = tp / (tp + fn + relevant_docs - tp)

        if rank >= max_steps:
            # tn = non_relevant_docs - fp
            # fn = relevant_docs - tp
            precision = tp / (tp + fp)
            # tnr = tn / (tn + fp)
            break
    return precision, recall

def tnr_at_recall_for_query(
    rankings: dict[str, float], qrels: dict[str, int], recall_level: float = 0.95
) -> float:
    #Normalized WSS
    tnr = 0
    tp, fp, tn, fn = 0, 0, 0, 0

    relevant_docs = sum(qrels.values())
    total_docs = len(qrels)
    non_relevant_docs = total_docs - relevant_docs

    for rank, (doc_id, _) in enumerate(rankings.items()):
        if qrels.get(doc_id, 0) == 1:  # Document is relevant
            tp += 1
        else:
            fp += 1

        recall = tp / (tp + fn + relevant_docs - tp)

        if recall >= recall_level:
            tn = non_relevant_docs - fp
            fn = relevant_docs - tp
            tnr = tn / (tn + fp)
            break

    return tnr

def wss_at_recall_for_query(
    rankings: dict[str, float], qrels: dict[str, int], recall_level: float = 0.95
) -> float:
    #Normal WSS
    tnr = 0
    tp, fp, tn, fn = 0, 0, 0, 0

    relevant_docs = sum(qrels.values())
    total_docs = len(qrels)
    non_relevant_docs = total_docs - relevant_docs

    for rank, (doc_id, _) in enumerate(rankings.items()):
        if qrels.get(doc_id, 0) == 1:  # Document is relevant
            tp += 1
        else:
            fp += 1

        recall = tp / (tp + fn + relevant_docs - tp)

        if recall >= recall_level:
            tn = non_relevant_docs - fp
            fn = relevant_docs - tp
            #tnr = tn / (tn + fp)
            wss= (tn + fn) / total_docs
            wss=wss-(1-recall_level)

            break

    return wss

def find_last_relevant_for_query(
    rankings: dict[str, float], qrels: dict[str, int]
) -> float:
    relevant_docs = [doc_id for doc_id, rel in qrels.items() if rel > 0]

    docs = rankings

    # Find the position of the last relevant document
    last_relevant_position = None
    for doc_id, _ in reversed(docs.items()):
        if doc_id in relevant_docs:
            last_relevant_position = (
                list(docs.keys()).index(doc_id) + 1
            )  # Adding 1 as indexing starts from 0
            break

    # If a relevant document is found in the run
    if last_relevant_position is not None:
        lr = last_relevant_position / len(docs) * 100

    if last_relevant_position is None:
        lr = 100.0

    return lr

def do_eval(path, recall_targets=[0.50, 0.75, 0.90, 0.95]):
    df=pd.read_csv(path)
    print(df.shape)


    rows=[]
    for c in df.columns:
        ranks = {str(i): float(i) for i, n in enumerate(df[c])}
        qrels = {str(i): int(content) for i, content in enumerate(df[c])}

        thisrow=[]
        thisrow.append(find_last_relevant_for_query(ranks, qrels))#find at which percentage of screening the last include was found

        wss=[]
        Nwss=[]
        precisionAtscreening=[]
        recallAtscreening=[]



        for target in recall_targets:

            Nwss.append(tnr_at_recall_for_query(ranks, qrels, recall_level=target))
            wss.append(wss_at_recall_for_query(ranks, qrels, recall_level=target))

            p,r= scores_at_percentage(ranks, qrels, completeness=target)
            precisionAtscreening.append(p)
            recallAtscreening.append(r)

        thisrow.extend(wss)
        thisrow.extend(Nwss)
        thisrow.extend(precisionAtscreening)
        thisrow.extend(recallAtscreening)
        rows.append(thisrow)

        print(statistics.mean(wss))
        print(statistics.stdev(wss))
        print("----")

    outdf = pd.DataFrame(rows,
                         columns=["Last Include", 'WSS@50r', 'WSS@75r', 'WSS@90r', 'WSS@95r', 'nWSS@50r', 'nWSS@75r',
                                  'nWSS@90r', 'nWSS@95r', 'precision@50%', 'precision@75%', 'precision@90%',
                                  'precision@95%', 'recall@50%', 'recall@75%', 'recall@90%', 'recall@95%'])
    return outdf

