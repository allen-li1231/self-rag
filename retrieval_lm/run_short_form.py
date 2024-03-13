import spacy
import jsonlines
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList
)
import random
import torch
import torch.nn.functional as F
import os
import numpy as np
import json
import argparse
import ast
import re
from tqdm import tqdm
from collections import Counter
import string
import sys
import time
from src.utils import DEVICE
from utils import PROMPT_DICT, TASK_INST, load_jsonlines, control_tokens, load_special_tokens
from metrics import match, accuracy


seed = 633

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def model_generate(prompt, model,
                   tokenizer=None,
                   temperature=0.8,
                   max_new_tokens=1024,
                   top_k=1,
                   top_p=1.,
                   beam_width=1,
                   do_sample=False,
                   num_return_sequences=1
):

    prompt = [prompt] if isinstance(prompt, str) else prompt
    if DEVICE == "mps":
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model.name_or_path,)
        inputs = tokenizer(prompt, padding="longest", return_tensors="pt").to(DEVICE)
        if temperature <= 0.:
            preds = model.generate(
                **inputs,
                top_p=top_p,
                num_beams=beam_width,
                temperature=None,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                output_scores=True,
                return_dict_in_generate=True)
        else:
            preds = model.generate(
                **inputs,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_beams=beam_width,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                output_scores=True,
                return_dict_in_generate=True)

        pred_token_ids = preds.sequences[:, inputs.input_ids.shape[1]:].to("cpu").numpy()
        pred_text = tokenizer.batch_decode(pred_token_ids)
        pred_log_probs = F.log_softmax(torch.stack(preds.scores), dim=2)
        pred_log_probs = torch.swapaxes(pred_log_probs, 0, 1)

    else:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_new_tokens,
            use_beam_search=True, n=num_return_sequences, logprobs=32016)
        preds = model.generate(prompt, sampling_params)
        pred_token_ids = [[output.token_ids for output in p.outputs[: num_return_sequences]] for p in preds]
        pred_text = [[output.text for output in p.outputs[: num_return_sequences]] for p in preds]
        pred_log_probs = [[output.logprobs for output in p.outputs[: num_return_sequences]] for p in preds]

    return pred_text, pred_token_ids, pred_log_probs


def postprocess_answer_option_conditioned(answer):
    for token in control_tokens:
        answer = answer.replace(token, "")

    if "</s>" in answer:
        answer = answer.replace("</s>", "")
    if "\n" in answer:
        answer = answer.replace("\n", "")

    if "<|endoftext|>" in answer:
        answer = answer.replace("<|endoftext|>", "")

    return answer


def call_model_rerank_w_scores_batch(prompt, evidences, model, max_new_tokens=15,
                                     ret_tokens=None, rel_tokens=None, grd_tokens=None, ut_tokens=None,
                                     use_seqscore=False, threshold=0.5, beam_width=2,
                                     w_rel=1.0, w_sup=1.0, w_use=0.5, mode="adaptive_retrieval", closed=False):
    results = {}
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path, padding="longest", padding_side="left") \
                if DEVICE == "mps" else None
        
    if mode != "always_retrieve":
        pred_text, pred_token_ids, pred_log_probs = model_generate(
            prompt, model, tokenizer=tokenizer, max_new_tokens=max_new_tokens,
            temperature=0., top_p=1., beam_width=beam_width, num_return_sequences=1
        )
        results["no_retrieval"] = pred_text[0]
        pred_token_ids = pred_token_ids[0]
        pred_log_probs = pred_log_probs[0]

    # save relevance token scores
    if mode == "always_retrieve":
        do_retrieve = True

    elif mode == "no_retrieval":
        do_retrieve = False

    else:
        if threshold is not None:
            score_dict = {}
            for tok, id in ret_tokens.items():
                if id not in pred_log_probs[0]:
                    score_dict[tok] = -100
                prob = pred_log_probs[0][id]
                score_dict[tok] = float(prob)
            do_retrieve = score_dict["[Retrieval]"] / (
                score_dict["[Retrieval]"] + score_dict["[No Retrieval]"]) > threshold
        else:
            do_retrieve = "[Retrieval]" in pred_text

    if do_retrieve is True:
        evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(
            para["title"], para["text"]) for para in evidences]
        lst_pred_text, lst_pred_token_ids, lst_pred_log_probs = model_generate(
            evidence_augmented_inputs, model, tokenizer=tokenizer, max_new_tokens=max_new_tokens,
            temperature=0., top_p=1., beam_width=beam_width, num_return_sequences=1
        )
        relevance_score_dict = {}
        grd_score_dict = {}
        ut_score_dict = {}
        overall_scores = {}
        for p_idx, (pred_text, pred_token_ids, pred_log_probs) \
            in enumerate(zip(lst_pred_text, lst_pred_token_ids, lst_pred_log_probs)):
            cumulative_logprob = pred_log_probs.sum()
            seq_score = cumulative_logprob / max(len(pred_log_probs), 1)

            relevance_score_dict.setdefault(p_idx, {})
            grd_score_dict.setdefault(p_idx, {})
            ut_score_dict.setdefault(p_idx, {})
            # Compute reward scores
            for tok, id in rel_tokens.items():
                prob = pred_log_probs[0][id]
                relevance_score_dict[p_idx][tok] = np.exp(float(prob))

            if grd_tokens is not None:
                idx = -1
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in grd_tokens.values():
                        idx = tok_idx
                        break
                if idx >= 0:
                    for token, token_id in grd_tokens.items():
                        prob = pred_log_probs[idx][token_id]
                        grd_score_dict[p_idx][token] = np.exp(float(prob))

            if ut_tokens is not None:
                idx = -1
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in ut_tokens.values():
                        idx = tok_idx
                        break
                if idx >= 0:
                    for token, token_id in ut_tokens.items():
                        prob = pred_log_probs[idx][token_id]
                        ut_score_dict[p_idx][token] = np.exp(float(prob))

            relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (
                np.sum(list(relevance_score_dict[p_idx].values())))

            if len(grd_score_dict[p_idx]) == 3:
                gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
                ground_score = (grd_score_dict[p_idx]["[Fully supported]"] / gt_sum) + 0.5 * (
                    grd_score_dict[p_idx]["[Partially supported]"] / gt_sum)
            else:
                ground_score = 0.0

            if len(ut_score_dict[p_idx]) == 5:
                ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
                ut_scores = [-1, -0.5, 0, 0.5, 1]
                utility_score = np.sum(
                    [ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i+1)] / ut_sum) for i in range(len(ut_scores))])
            else:
                utility_score = 0.0

            if use_seqscore is True:
                final_score = np.exp(seq_score) + w_rel * relevance_score + \
                    w_sup * ground_score + w_use * utility_score
            else:
                final_score = w_rel * relevance_score + \
                    w_sup * ground_score + w_use * utility_score

            overall_scores[p_idx] = {"final_score": final_score,
                                     "relevance_score": relevance_score,
                                     "ground_score": ground_score,
                                     "utility_score": utility_score,
                                     "relevance_score_dict": relevance_score_dict,
                                     "grd_score_dict": grd_score_dict,
                                     "ut_score_dict": utility_score}
            results["retrieval_{}".format(p_idx)] = {
                "pred": pred_text, "score": final_score, "ctx": evidences[p_idx]}

    else:
        prompt += "[No Retrieval]"
        pred, pred_token_ids, pred_log_probs = model_generate(
            prompt, model, tokenizer=tokenizer, max_new_tokens=max_new_tokens,
            temperature=0., top_p=1., beam_width=beam_width, num_return_sequences=1
        )

    # Aggregating answers
    if len(results) == 1:
        postprocessed_pred = postprocess_answer_option_conditioned(pred)
        return postprocessed_pred, results, do_retrieve
    else:
        answer2score = {}
        if closed is True:
            for key, result in results.items():
                if key == "no_retrieval":
                    continue
                answer = postprocess_answer_option_conditioned(result["pred"])
                score = result["score"]
                answer2score.setdefault(answer, 0)
                answer2score[answer] += score
            sorted_answers = sorted(
                answer2score.items(), key=lambda x: x[1], reverse=True)
            best_option = sorted_answers[0][0]
        else:
            path2score = {key: item["score"] for key,
                          item in results.items() if key != "no_retrieval"}
            best_path = sorted(path2score.items(),
                               key=lambda x: x[1], reverse=True)[0][0]
            best_option = results[best_path]["pred"]
        return best_option, results, do_retrieve


def process_data_evidences(demonstration, top_n):
    ctx_key = "ctxs" if "ctxs" in demonstration else "top_contexts"
    prompt = PROMPT_DICT["prompt_no_input"].format_map(demonstration)
    evidences = demonstration[ctx_key][:top_n]
    return prompt, evidences


def preprocess_input_data(dataset, task=None):
    new_data = []
    if task in TASK_INST:
        instruction = TASK_INST[task]
    else:
        instruction = None
    for item in dataset:
        if task == "arc_c":
            choices = item["choices"]
            answer_labels = {}
            for i in range(len(choices["label"])):
                answer_key = choices["label"][i]
                text = choices["text"][i]
                if answer_key == "1":
                    answer_labels["A"] = text
                if answer_key == "2":
                    answer_labels["B"] = text
                if answer_key == "3":
                    answer_labels["C"] = text
                if answer_key == "4":
                    answer_labels["D"] = text
                if answer_key in ["A", "B", "C", "D"]:
                    answer_labels[answer_key] = text

            if "D" not in answer_labels:
                answer_labels["D"] = ""
            choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
                answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
            if "E" in answer_labels:
                choices += "\nE: {}".format(answer_labels["E"])
            item["instruction"] = instruction + \
                "\n\n### Input:\n" + item["question"] + choices
            item["answers"] = [item["answerKey"]]
        else:
            prompt = instruction + "\n\n## Input:\n\n" + \
                item["question"] if instruction is not None else item["question"]
            item["instruction"] = prompt
        new_data.append(item)

    return new_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--download_dir', type=str, help="specify vllm model download dir",
                        default=".cache")
    parser.add_argument("--ndocs", type=int, default=10,
                        help="Number of documents to retrieve per questions")
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--dtype",  type=str, default="half",
                        help="We use bfloat16 for training. If you run inference on GPUs that do not support BF16, please set this to be `half`.")
    # Decoding hyperparams
    parser.add_argument('--threshold', type=float, 
                        default=None, help="Adaptive threshold.")
    parser.add_argument("--use_seqscore", action="store_true")
    parser.add_argument("--use_groundness", action="store_true",
                        help="use ground score")
    parser.add_argument(
        "--use_utility", action="store_true", help="tree search")
    parser.add_argument("--beam_width",  type=int,
                        default=2, help="beam search width")
    parser.add_argument("--w_rel",  type=float, default=1.0,
                        help="reward weight for document relevance")
    parser.add_argument("--w_sup",  type=float, default=1.0,
                        help="reward weight for generation support (attribution)")
    parser.add_argument("--w_use",  type=float, default=0.5,
                        help="reward weight for overall completeness / utility.")
    parser.add_argument('--mode', type=str, help="mode to control retrieval.",
                        default="default", choices=['adaptive_retrieval', 'no_retrieval', 'always_retrieve'],)
    parser.add_argument('--metric', type=str, help="metric to be used during evaluation")
    args = parser.parse_args()
    gpt = args.model_name
    input_path = args.input_file
    if input_path.endswith(".json"):
        input_data = json.load(open(input_path))
    else:
        input_data = load_jsonlines(input_path)

    input_data = preprocess_input_data(
        input_data, task=args.task)
    tokenizer = AutoTokenizer.from_pretrained(gpt, padding="longest", padding_side="left")
    if DEVICE == 'mps':
        model = AutoModelForCausalLM.from_pretrained(gpt, device_map=DEVICE)
        model.eval()
    else:
        from vllm import LLM
        if args.dtype is not None:
            model = LLM(model=gpt, download_dir=args.download_dir,
                        dtype=args.dtype, tensor_parallel_size=args.world_size,)
        else:
            model = LLM(model=gpt, download_dir=args.download_dir,
                        dtype=args.dtype, tensor_parallel_size=args.world_size,)

    # Get token ids for reflection tokens.
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=args.use_groundness, use_utility=args.use_utility)

    def generate(prompt, evidences, max_new_tokens):
        return call_model_rerank_w_scores_batch(prompt, evidences=evidences, model=model, max_new_tokens=max_new_tokens,
                                                rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                                beam_width=args.beam_width, threshold=args.threshold, use_seqscore=args.use_seqscore,
                                                w_rel=args.w_rel, w_sup=args.w_sup, w_use=args.w_use, mode=args.mode, closed=args.task in ["fever", "arc_c"])

    preds = []
    prompts = []
    golds = []
    metric_results = []
    scores = []
    all_results = []
    count = 0
    for i, row in tqdm(enumerate(input_data)):
        results = {}
        prompt = PROMPT_DICT["prompt_no_input"].format_map(row)
        _, evidences = process_data_evidences(row, top_n=args.ndocs)
        pred, results, do_retrieve = generate(
            prompt, evidences, max_new_tokens=args.max_new_tokens,)
        if type(pred) is str and pred[0] == "#" or pred[0] == ":":
            pred = pred[1:]
        prompts.append(prompt)
        preds.append(pred)
        all_results.append(results)
        if do_retrieve is True:
            count += 1
        if "answers" not in row and "answer" in row:
            row["answers"] = [row["answer"]] if type(
                row["answer"]) is str else row["answer"]
        if args.metric == "accuracy":
            metric_result = accuracy(pred, row["output"])

        elif args.metric == "match":
            if "SUPPORTS" in pred:
                pred = "true"
            elif "REFUTES" in pred:
                pred = "false"
            metric_result = match(pred, row["answers"])
        else:
            raise NotImplementedError

        metric_results.append(metric_result)
        if i % 10 == 0:
            print("average: {}".format(np.mean(metric_results)))
            final_results = {"preds": preds, "prompts": prompts, "metric_results": metric_results, "all_results": all_results,
                             "golds": golds,  "metric":  args.metric, "metric_mean": np.mean(metric_results), "scores": scores}
            with open(args.output_file + "_tmp", "w") as outfile:
                json.dump(final_results, outfile)

    final_results = {"preds": preds, "prompts": prompts, "metric_results": metric_results, "all_results": all_results,
                     "golds": golds,  "metric":  args.metric, "metric_mean": np.mean(metric_results), "scores": scores}
    with open(args.output_file, "w") as outfile:
        json.dump(final_results, outfile)

    print("Final result: {0}".format(np.mean(metric_results)))
    print("Retrieval Frequencies: {0}".format(count / len(final_results)))


if __name__ == "__main__":
    main()
