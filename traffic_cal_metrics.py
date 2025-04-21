"""
eval_metrics_norm.py
计算 BLEU‑4 / Rouge‑L.precision / MoverScore / BERTScore(F1)
GT : sft_formatted_*.json
Pred: 每行 {video, type, question, response}
使用方式：
conda env create -f traffic_eval.yml
conda activate traffic_eval
python eval_metrics_norm.py \
  --gt path/to/sft_formatted_test.json \
  --pred path/to/traffic_results.jsonl \
  --out path/to/metric_results.jsonl \
  --skip_time_window

# omni
LD_LIBRARY_PATH=/mnt/cache/share/gcc/gcc-11.2.0/lib64:$LD_LIBRARY_PATH \
LD_PRELOAD=/mnt/cache/share/gcc/gcc-11.2.0/lib64/libstdc++.so.6 \
srun python traffic_cal_metrics.py \
  --gt ../datasets/AV-TAU-R1/annotations/sft_formatted_test.json \
  --pred ./traffic_results_Qwen2.5-Omni-7B.jsonl \
  --out ./test_metric_results.jsonl \
  --skip_time_window

"""
import argparse, json, os, time, numpy as np, torch, jieba
from collections import defaultdict, Counter
from transformers import logging
logging.set_verbosity_error()

# ---------- 指标函数 ----------
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
_bleu_smooth = SmoothingFunction().method7
def bleu(r, h): return sentence_bleu([list(jieba.cut(r))], list(jieba.cut(h)),
                                     weights=(.25,.25,.25,.25), smoothing_function=_bleu_smooth)

from rouge_score import rouge_scorer
rouger = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
def rougel(r, h): return rouger.score(r, h)["rougeL"].precision

from moverscore_v2 import word_mover_score
from collections import defaultdict as dd
def moverscore(r, h): return np.mean(word_mover_score([r],[h], dd(lambda:1.), dd(lambda:1.),
                                                     stop_words=[], n_gram=1, remove_subwords=False))

from bert_score import score as bert_score
def bertscore(r, h):
    _,_,f = bert_score([h],[r], model_type="roberta-large",
                       device="cuda:0" if torch.cuda.is_available() else "cpu",
                       lang="en", verbose=False)
    return f[0].item()

METRIC_FUNCS = dict(bleu=bleu, rouge=rougel, moverscore=moverscore, bertscore=bertscore)
TASKS        = ["description", "reason", "response", "prevention"]

# ---------- 工具 ----------
def norm_video_path(p:str)->str:
    """返回 videos/... 形式"""
    idx = p.find("videos/")
    return p[idx:] if idx!=-1 else p

def norm_task_name(t:str)->str:
    return "description" if t=="discription" else t

def parse():
    pa = argparse.ArgumentParser()
    pa.add_argument("--gt",   required=True)
    pa.add_argument("--pred", required=True)
    pa.add_argument("--out",  required=True)
    pa.add_argument("--skip_time_window", action="store_true")
    return pa.parse_args()

# ---------- 主流程 ----------
def main():
    args = parse()

    # 读 GT
    gt_map = {}
    for item in json.load(open(args.gt)):
        if args.skip_time_window and item["type"]=="time_window": continue
        vid  = norm_video_path(item["video"])
        task = norm_task_name(item["type"])
        key  = f"{vid}--{task}"
        gt_map[key] = item["QA"][0]["a"]

    # 已评集合
    done=set()
    if os.path.exists(args.out):
        for ln in open(args.out): done.update(json.loads(ln).keys())

    # 汇总
    sum_m = defaultdict(float); cnt = Counter()
    start=time.time()

    with open(args.out,"a") as fout, open(args.pred) as fin:
        for i,line in enumerate(fin,1):
            pr = json.loads(line)
            vid  = norm_video_path(pr["video"])
            task = norm_task_name(pr["type"])
            key  = f"{vid}--{task}"
            if key in done or key not in gt_map: continue

            ref, hyp = gt_map[key], pr["response"]
            per = {m:f(ref,hyp) for m,f in METRIC_FUNCS.items()}
            fout.write(json.dumps({key:per},ensure_ascii=False)+"\n"); fout.flush()

            for m,v in per.items():
                sum_m[(task,m)] += v; sum_m[("all",m)] += v
            cnt[task]+=1; cnt["all"]+=1

            if i%50==0:
                print(f"[{i}] elapsed {(time.time()-start)/60:.1f} min")

    # 打印矩阵
    print("\n===== Average scores =====")
    head=["metric"]+TASKS+["overall"]
    print("{:<12}".format(head[0])+ "".join(f"{h:>12}" for h in head[1:]))
    for m in METRIC_FUNCS:
        row = [m]
        for t in TASKS+["all"]:
            row.append(f"{(sum_m[(t,m)]/cnt[t]):.4f}" if cnt[t] else "  n/a  ")
        print("{:<12}".format(row[0])+ "".join(f"{c:>12}" for c in row[1:]))

if __name__ == "__main__":
    main()
