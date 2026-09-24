"""Evaluate token BIO F1 and exact intent-span recovery on the held-out test split."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from datasets import Dataset
from seqeval.metrics import f1_score
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer

ROOT=Path(__file__).resolve().parents[1]; LABELS=["O","B-INTENT","I-INTENT"]
def rows(path): return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x]
def spans(tags):
 out=[]; start=None
 for i,t in enumerate(tags+["O"]):
  if t=="B-INTENT":
   if start is not None: out.append((start,i-1))
   start=i
  elif t!="I-INTENT" and start is not None: out.append((start,i-1)); start=None
 return out
def main():
 data=rows(ROOT/'data'/'splits'/'test.jsonl'); tokenizer=AutoTokenizer.from_pretrained(ROOT/'model'/'checkpoints'/'best')
 ds=Dataset.from_list(data)
 def encode(batch): return tokenizer(batch['tokens'],is_split_into_words=True,truncation=True,max_length=128)
 encoded=ds.map(encode,batched=True,remove_columns=ds.column_names)
 model=AutoModelForTokenClassification.from_pretrained(ROOT/'model'/'checkpoints'/'best')
 prediction=Trainer(model=model,tokenizer=tokenizer,data_collator=DataCollatorForTokenClassification(tokenizer)).predict(encoded)
 true,pred=[],[]; exact=defaultdict(lambda:[0,0]); all_true=[]; all_pred=[]; prediction_rows=[]
 for row,logits in zip(data,prediction.predictions):
  word_ids=tokenizer(row['tokens'],is_split_into_words=True,truncation=True,max_length=128).word_ids()
  values=[]; seen=set()
  for wi,vector in zip(word_ids,logits):
   if wi is not None and wi not in seen: values.append(LABELS[int(np.argmax(vector))]); seen.add(wi)
  actual=row['bio_tags'][:len(values)]; values=values[:len(actual)]
  true.append(actual); pred.append(values); all_true.append(spans(actual)); all_pred.append(spans(values))
  prediction_rows.append({'utterance_id':row['utterance_id'],'reference_spans':spans(actual),'bert_spans':spans(values)})
  bucket=exact[row['num_intents']]; bucket[1]+=1; bucket[0]+=int(spans(actual)==spans(values))
 total=sum(x[1] for x in exact.values()); correct=sum(x[0] for x in exact.values())
 report={'boundary_source':'source_corpus_exact_match_v1','token_bio_f1':f1_score(true,pred),'exact_intent_set_accuracy':correct/total,'examples':total,'by_num_intents':{str(k):{'exact_accuracy':v[0]/v[1],'correct':v[0],'examples':v[1]} for k,v in exact.items()}}
 (ROOT/'results'/'metrics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,indent=2))
 (ROOT/'results'/'bert_test_predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in prediction_rows),encoding='utf-8')
if __name__=='__main__': main()
