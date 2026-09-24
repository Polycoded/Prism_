"""Compare exact-source BERT boundaries to the parent rule splitter as a subprocess."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WORKER="""import sys,json,os
os.environ['CITEFRONTIER_PARSER']='rules'
from prototype.runtime import LiveRuntime
r=LiveRuntime(parser='rules')
for line in sys.stdin:
 x=json.loads(line); out=[]
 try:
  for item in r.decomposer.split(x['text']): out.extend(item.public().get('source_spans',[]))
  print(json.dumps({'spans':out}))
 except Exception as e: print(json.dumps({'error':str(e)}))
 sys.stdout.flush()
"""
def rows(): return [json.loads(x) for x in (ROOT/'data'/'splits'/'test.jsonl').read_text(encoding='utf-8').splitlines() if x]
def spans(tags):
 out=[]; start=None
 for i,t in enumerate(tags+['O']):
  if t=='B-INTENT':
   if start is not None: out.append((start,i-1))
   start=i
  elif t!='I-INTENT' and start is not None: out.append((start,i-1)); start=None
 return out
def char_to_tokens(text, char_spans):
 words=[]; pos=0
 for i,t in enumerate(text.split()):
  s=text.find(t,pos); words.append((s,s+len(t))); pos=s+len(t)
 out=[]
 for a,b in char_spans:
  ids=[i for i,(s,e) in enumerate(words) if s>=a and e<=b]
  if ids: out.append((ids[0],ids[-1]))
 return out
def main():
 data=rows(); parent=ROOT.parent; bert={row['utterance_id']:row for row in [json.loads(x) for x in (ROOT/'results'/'bert_test_predictions.jsonl').read_text(encoding='utf-8').splitlines() if x]}
 p=subprocess.Popen([sys.executable,'-c',WORKER],cwd=parent,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
 rule_correct=0; samples={'rule_correct_bert_wrong':[],'bert_correct_rule_wrong':[],'both_wrong':[]}
 # BERT metric already computed; its exact span predictions are in the validated test result.
 # Compare rules to deterministic recovered reference; BERT's test exact accuracy is reported separately.
 for row in data:
  text=' '.join(row['tokens']); p.stdin.write(json.dumps({'text':text})+'\n'); p.stdin.flush(); result=json.loads(p.stdout.readline())
  reference=spans(row['bio_tags']); predicted=char_to_tokens(text,result.get('spans',[])); rule_ok=predicted==reference; bert_predicted=[tuple(x) for x in bert[row['utterance_id']]['bert_spans']]; bert_ok=bert_predicted==reference; rule_correct+=rule_ok
  category='rule_correct_bert_wrong' if rule_ok and not bert_ok else 'bert_correct_rule_wrong' if bert_ok and not rule_ok else 'both_wrong' if not bert_ok and not rule_ok else None
  if category and len(samples[category])<12: samples[category].append({'utterance':text,'reference':reference,'rule_spans':predicted,'bert_spans':bert_predicted})
 p.stdin.close(); p.wait(timeout=30)
 metrics=json.loads((ROOT/'results'/'metrics.json').read_text(encoding='utf-8'))
 rule_acc=rule_correct/len(data)
 text=['# Rule-based comparison','','Both systems were evaluated against `source_corpus_exact_match_v1` recovered test boundaries.','',f"- BERT exact intent-set accuracy: {metrics['exact_intent_set_accuracy']:.6f}",f'- Rule-based exact intent-set accuracy: {rule_acc:.6f}', '', '## BERT correct, rule-based wrong']
 for sample in samples['bert_correct_rule_wrong']:
  text.extend([f"- `{sample['utterance']}`",f"  - reference: {sample['reference']}; rule: {sample['rule_spans']}"])
 for title,key in [('Rule-based correct, BERT wrong','rule_correct_bert_wrong'),('Both wrong','both_wrong')]:
  text.extend(['',f'## {title}'])
  if samples[key]:
   for sample in samples[key]: text.extend([f"- `{sample['utterance']}`",f"  - reference: {sample['reference']}; BERT: {sample['bert_spans']}; rule: {sample['rule_spans']}"])
  else: text.append('- No examples in the retained sample set.')
 (ROOT/'results'/'error_analysis.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
 (ROOT/'results'/'rule_based_comparison.json').write_text(json.dumps({'boundary_source':'source_corpus_exact_match_v1','examples':len(data),'rule_based_exact_accuracy':rule_acc,'bert_exact_accuracy':metrics['exact_intent_set_accuracy']},indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'rule_based_exact_accuracy':rule_acc,'examples':len(data)},indent=2))
if __name__=='__main__': main()
