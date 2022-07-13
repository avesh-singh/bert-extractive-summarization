def calc_scores(pred_summaries, gold_summaries,
                      keys=['rouge1', 'rougeL'], use_stemmer=True):
    # Calculate rouge scores
    scorer = rouge_scorer.RougeScorer(keys, use_stemmer=use_stemmer)
    n = len(pred_summaries)
    scores = [scorer.score(pred_summaries[j], gold_summaries[j]) for
              j in range(n)]

    # create dict
    dict_scores = {}
    for key in keys:
        dict_scores.update({key: {}})

    # populate dict
    for key in keys:
        precision_list = [scores[j][key][0] for j in range(len(scores))]
        recall_list = [scores[j][key][1] for j in range(len(scores))]
        f1_list = [scores[j][key][2] for j in range(len(scores))]

        precision = np.mean(precision_list)
        recall = np.mean(recall_list)
        f1 = np.mean(f1_list)

        dict_results = {'recall': recall, 'precision': precision, 'f1': f1}

        dict_scores[key] = dict_results

    return dict_scores