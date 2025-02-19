#!/usr/bin/env python

import json
from collections import Counter
import string
import re
import sys
import os


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    if ground_truth is None or prediction is None:
        return exact_match_score(prediction, ground_truth)
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    if ground_truth is None or prediction is None:
        return prediction == ground_truth
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(gold_answers, predictions, out_res_file):
    f1 = exact_match = total = 0.

    for ground_truths, prediction in zip(gold_answers, predictions):
        total += 1
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    with open(out_res_file, "w") as f:
        f.write("EM: %.4f" % exact_match + "\n")
        f.write("F1: %.4f" % f1 + "\n")


def load_ans_json(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


def get_gold_pred(gold_json_file, pred_json_file):
    gold_dict = load_ans_json(gold_json_file)
    pred_dict = load_ans_json(pred_json_file)
    gold_ans = []
    pred_ans = []
    for rid in gold_dict:
        for qid in gold_dict[rid]:
            gold_ans.append(gold_dict[rid][qid])
            pred_ans.append(pred_dict[rid][qid])
    assert len(gold_ans) == len(pred_ans), "the length of gold and pred does not match!"
    return gold_ans, pred_ans


if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    submit_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(submit_dir):
        print("%f doesn't exist" % submit_dir)

    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    gold, pred = get_gold_pred(os.path.join(truth_dir, 'gold.json'), os.path.join(submit_dir, "r2vq_pred.json"))
    evaluate(gold, pred, os.path.join(output_dir, 'scores.txt'))
