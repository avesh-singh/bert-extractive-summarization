from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
# data_directory = "DUCDatasets-20220708T144932Z-001/DUCDatasets/DUC2007_Summarization_Documents"
# summary_directory = data_directory + "/mainEval/ROUGE"
#
# model_summary_file = summary_directory + '/models/D0701.M.250.A.I'
# peer_summary_file = "results/summary.txt"

data_directory = "DUC2006_Summarization_Documents"
summary_directory = data_directory + "/NISTeval/ROUGE"

model_summary_file = summary_directory + '/models/D0601.M.250.A.E'
peer_summary_file = "results/summary.txt"

with open(peer_summary_file, 'r') as file:
    summaries = file.read()

with open(model_summary_file, 'r') as file:
    model_summary = file.read()

print(summaries)
# model_summary = '''Craig Eccleston-Todd, 27, had drunk at least three pints before driving car . Was using phone when he veered across road in Yarmouth, Isle of Wight . Crashed head-on into 28-year-old Rachel Titley's car, who died in hospital . Police say he would have been over legal drink-drive limit at time of crash . He was found guilty at Portsmouth Crown Court of causing death by dangerous driving .'''
# summaries = '''A drunk driver who killed a young woman in a head-on crash while checking his mobile phone has been jailed for six years. Craig Eccleston-Todd, 27, was driving home from a night at a pub when he received a text message. Mr Eccleston-Todd's car was barely recognisable (pictured) Police said Eccleston-Todd had drunk at least three or four pints of beer before getting behind the wheel. He was found guilty of causing death by dangerous driving at Portsmouth Crown Court yesterday. Miss Titley’s death in these circumstances reiterates the danger of using a hand-held mobile phone whilst driving.’ Eccleston-Todd was found guilty of causing death by dangerous driving following a trial at Portsmouth Crown Court (pictured) He added: 'Mr Eccleston-Todd will now spend six years behind bars, but Rachel's family have lost her forever. ' Those who continue to do so risk spending a substantial time in prison.'''
scores = scorer.score(model_summary, summaries)
print(scores)