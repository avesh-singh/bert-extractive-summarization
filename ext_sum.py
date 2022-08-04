import time
import numpy as np
import torch
from transformers import BertTokenizer
import pickle
import torch.nn as nn
import torch.nn.functional as F


def preprocess(article):
    processed_text = ["[CLS] " + sentence + "[SEP] " for sentence in article]
    return "".join(processed_text)


def load_text(processed_text, max_pos, device):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    sep_vid = tokenizer.vocab["[SEP]"]
    cls_vid = tokenizer.vocab["[CLS]"]

    def _process_src(raw):
        raw = raw.strip().lower()
        raw = raw.replace("[cls]", "[CLS]").replace("[sep]", "[SEP]")
        src_subtokens = tokenizer.tokenize(raw)
        # src_subtokens = ["[CLS]"] + src_subtokens + ["[SEP]"]
        src_subtoken_idxs = tokenizer.convert_tokens_to_ids(src_subtokens)
        if len(src_subtoken_idxs) < max_pos:
            src_subtoken_idxs.extend([0] * (max_pos - len(src_subtoken_idxs) + 1))
        src_subtoken_idxs = src_subtoken_idxs[:-1][:max_pos]
        src_subtoken_idxs[-1] = sep_vid
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        
        segments_ids = []
        segs = segs[:max_pos]
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        src = torch.tensor(src_subtoken_idxs)[None, :].to(device)
        mask_src = (1 - (src == 0).float()).to(device)
        cls_ids = [[i for i, t in enumerate(src_subtoken_idxs) if t == cls_vid]]
        clss = torch.tensor(cls_ids).to(device)
        mask_cls = 1 - (clss == -1).float()
        clss[clss == -1] = 0
        return src, mask_src, segments_ids, clss, mask_cls

    src, mask_src, segments_ids, clss, mask_cls = _process_src(processed_text)
    segs = torch.tensor(segments_ids)[None, :].to(device)
    src_text = [[sent.replace("[SEP]", "").strip() for sent in processed_text.split("[CLS]")]]
    return src, mask_src, segs, clss, mask_cls, src_text


def test(model, input_data, result_path, max_length, write_to_file=False):
    def _generate_pred():
        with torch.no_grad():
            src, mask, segs, clss, mask_cls, src_str = input_data
            sent_scores, mask = model(src, segs, clss, mask, mask_cls)
            convert_to_probability = nn.Sigmoid()
            sent_scores = convert_to_probability(sent_scores)[0]
            sent_scores = sent_scores + mask.float()
            sent_scores = sent_scores.cpu().data.numpy()
            selected_ids = np.argsort(-sent_scores, 1)

            pred = []
            for i, idx in enumerate(selected_ids):
                _pred = []
                if len(src_str[i]) == 0:
                    continue
                for j in selected_ids[i][: len(src_str[i])]:
                    if j >= len(src_str[i]):
                        continue
                    candidate = src_str[i][j].strip()
                    _pred.append(candidate)
                    if len(_pred) == max_length:
                        break
                _pred = " ".join(_pred)
                pred.append(_pred)
        return pred

    if write_to_file:
        with open(result_path, "w") as save_pred:
            pred = _generate_pred()
            for i in range(len(pred)):
                save_pred.write(pred[i].strip() + "\n")
    else:
        pred = "\n".join(_generate_pred())
    return pred


def summarize(sample, result_fp, model, max_length=3, max_pos=512, return_summary=True, write_summary=False):
    model.eval()
    processed_text = preprocess(sample['sentences'])
    input_data = load_text(processed_text, max_pos, device="cpu")
    predicted = test(model, input_data, result_fp, max_length, write_summary)
    if return_summary:
        return predicted

# def summarize(input_fp, result_fp, model, max_length=3, max_pos=512, return_summary=True):
#     model.eval()
#     with open(input_fp, 'rb') as file:
#         contents = pickle.load(file)
#     for i, sample in enumerate(contents):
#         processed_text = preprocess(sample['sentences'])
#         input_data = load_text(processed_text, max_pos, device="cpu")
#         test(model, input_data, result_fp+f"_{i}", max_length, block_trigram=True)
#     if return_summary:
#         return open(result_fp).read().strip()



