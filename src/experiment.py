import argparse,base64,csv,hashlib,io,json,os,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROMPT='''Classify the largest, primary aircraft in the photo by the primary intended role of its variant.
military: designed primarily for armed forces, including unarmed trainers,
transports, tankers and patrol aircraft. civilian: designed primarily for passenger,
cargo, private, agricultural, firefighting or sport aviation. A military operator
alone does not change a civilian design's class. Choose the most likely class
from visual evidence. Return ONLY JSON with keys label (military or civilian)
and reason (short visual explanation). Do not use outside tools.'''
def read_csv(p):
 with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def prepare():
 import requests
 from PIL import Image,ImageOps
 manifest=ROOT/'data/manifest.csv'
 if manifest.exists():print('Manifest already frozen');return
 session=requests.Session();session.headers['User-Agent']='AircraftCourseProject/1.0 (educational image classification)'
 def get(url,**kw):
  for attempt in range(5):
   r=session.get(url,timeout=90,**kw)
   if r.status_code not in (429,503):r.raise_for_status();return r
   time.sleep(min(30,5*(attempt+1)))
  r.raise_for_status()
 def api(url,**kw):return get(url,params={'action':'query','format':'json',**kw}).json()['query']['pages']
 selection=read_csv(ROOT/'data/selection.csv');cache=ROOT/'data/source_cache.json'
 if cache.exists():sources=json.loads(cache.read_text(encoding='utf-8'))
 else:
  pages=api('https://en.wikipedia.org/w/api.php',titles='|'.join(r['page_title'] for r in selection),redirects=1,prop='pageimages',piprop='name|thumbnail',pithumbsize=960,pilicense='free')
  sources={p['title']:p for p in pages.values()}
  cache.write_text(json.dumps(sources,ensure_ascii=False),encoding='utf-8')
 cc=ROOT/'data/commons_cache.json'
 if cc.exists():commons=json.loads(cc.read_text(encoding='utf-8'))
 else:
  commons=api('https://commons.wikimedia.org/w/api.php',titles='|'.join('File:'+p['pageimage'] for p in sources.values()),prop='imageinfo',iiprop='url|extmetadata')
  cc.write_text(json.dumps(commons,ensure_ascii=False),encoding='utf-8')
 infos={p['title'].removeprefix('File:'):p['imageinfo'][0] for p in commons.values() if 'imageinfo' in p}
 rows=[];(ROOT/'data/images').mkdir(exist_ok=True)
 for row in selection:
  p=sources.get(row['page_title'])
  if p is None:
   match=[v for k,v in sources.items() if k.lower()==row['page_title'].lower()]
   if not match:raise RuntimeError('Redirect: adjust selection title '+row['page_title'])
   p=match[0]
  if 'thumbnail' not in p:raise RuntimeError('No free lead photo: '+row['page_title'])
  info=infos[p['pageimage'].replace('_',' ')];meta=info.get('extmetadata',{});field=lambda k:meta.get(k,{}).get('value','')
  if not field('LicenseShortName'):raise RuntimeError('Missing license')
  path=ROOT/'data/images'/(row['id']+'.jpg')
  if not path.exists():
   r=get(p['thumbnail']['source']);im=ImageOps.exif_transpose(Image.open(io.BytesIO(r.content))).convert('RGB');im.thumbnail((960,960));im.save(path,quality=90);time.sleep(1)
  rows.append({**row,'image_path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'article_url':'https://en.wikipedia.org/wiki/'+p['title'].replace(' ','_'),'file_url':info['descriptionurl'],'download_url':p['thumbnail']['source'],'author':field('Artist'),'license':field('LicenseShortName'),'license_url':field('LicenseUrl'),'description':field('ImageDescription'),'reviewed':'0'})
  print(row['id'],row['page_title'],flush=True)
 write_csv(manifest,rows)
def checked_rows():
 rows=read_csv(ROOT/'data/manifest.csv')
 assert len(rows)==30 and len({r['id'] for r in rows})==30
 assert sum(r['label']=='military' for r in rows)==15 and sum(r['label']=='civilian' for r in rows)==15
 assert all(r['reviewed']=='1' for r in rows),'Review every photo and set reviewed=1 first'
 hashes=[]
 for r in rows:
  h=hashlib.sha256((ROOT/r['image_path']).read_bytes()).hexdigest();assert h==r['sha256'];hashes.append(h)
 assert len(set(hashes))==30,'Duplicate photos'
 return rows
def parse_label(raw):
 d=json.loads(raw)
 if not isinstance(d,dict) or d.get('label') not in ('military','civilian'):raise ValueError('Invalid schema')
 return d['label']
def run(limit=None):
 from huggingface_hub import HfApi,InferenceClient
 rows=checked_rows();token=os.environ.get('HF_TOKEN')
 if not token:
  from huggingface_hub import get_token
  token=get_token()
 if not token:raise RuntimeError('Set HF_TOKEN securely')
 config=json.loads((ROOT/'config.json').read_text());assert len(config['models'])==2
 assert len({m['id'] for m in config['models']})==2
 for m in config['models']:
  info=HfApi(token=token).model_info(m['id'],expand=['inferenceProviderMapping'])
  mapping=info.inference_provider_mapping or []
  pairs=mapping.items() if isinstance(mapping,dict) else ((v.provider,v) for v in mapping)
  available={k:v for k,v in pairs if v.status=='live' and v.task in ('image-text-to-text','conversational')}
  if not available:raise RuntimeError('No live vision provider: '+m['id'])
  if m['provider']=='auto':m['provider']=sorted(available)[0]
  if m['provider'] not in available:raise RuntimeError('Provider unavailable')
 out=ROOT/'results';out.mkdir(exist_ok=True);log=out/'predictions.jsonl'
 signature=hashlib.sha256(json.dumps({'config':config,'prompt':PROMPT,'data':[(r['id'],r['sha256'],r['label'],r['hard']) for r in rows]},sort_keys=True).encode()).hexdigest()
 old=[json.loads(s) for s in log.read_text(encoding='utf-8').splitlines()] if log.exists() else []
 if any(r['experiment']!=signature for r in old):raise RuntimeError('Experiment changed: archive old results first')
 done={(r['model'],r['id']) for r in old};count=0
 budget=config['max_new_requests'] if limit is None else min(limit,config['max_new_requests'])
 clients={m['id']:InferenceClient(provider=m['provider'],api_key=token,timeout=120) for m in config['models']}
 for i,row in enumerate(rows):
  for m in config['models'][::1 if i%2==0 else -1]:
   if (m['id'],row['id']) in done:continue
   if count>=budget:print('Request limit reached; resume later');return
   encoded=base64.b64encode((ROOT/row['image_path']).read_bytes()).decode()
   messages=[{'role':'user','content':[{'type':'text','text':PROMPT},{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+encoded}}]}]
   rec={'experiment':signature,'id':row['id'],'model':m['id'],'provider':m['provider'],'truth':row['label'],'hard':int(row['hard']),'sha256':row['sha256'],'timestamp':datetime.now(timezone.utc).isoformat()}
   start=time.perf_counter();count+=1
   try:response=clients[m['id']].chat.completions.create(model=m['id'],messages=messages,temperature=config['temperature'],max_tokens=config['max_tokens'])
   except Exception as e:raise RuntimeError('HF request stopped ('+type(e).__name__+'); prior answers saved; no automatic retries') from None
   rec['latency_s']=time.perf_counter()-start;rec['raw']=response.choices[0].message.content or ''
   rec['usage']=dict(response.usage) if response.usage is not None else None
   try:rec['prediction']=parse_label(rec['raw']);rec['status']='ok'
   except (ValueError,TypeError):rec['prediction']='invalid';rec['status']='invalid'
   with log.open('a',encoding='utf-8') as f:f.write(json.dumps(rec,ensure_ascii=False,default=str)+'\n')
   print(row['id'],m['id'],rec['prediction']);time.sleep(.5)
def report():
 import pandas as pd
 import matplotlib.pyplot as plt
 from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
 rows=checked_rows();path=ROOT/'results/predictions.jsonl'
 if not path.exists():raise RuntimeError('No real predictions yet')
 df=pd.read_json(path,lines=True,dtype={'id':str})
 expected={m['id'] for m in json.loads((ROOT/'config.json').read_text())['models']}
 assert len(df)==60 and set(df.model)==expected,'Complete all 60 responses before comparing'
 assert not df.duplicated(['model','id']).any() and df.experiment.nunique()==1
 assert all(set(g.id)=={r['id'] for r in rows} for _,g in df.groupby('model'))
 metrics=[];fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,(model,g) in zip(axes,df.groupby('model')):
  p,r,f,_=precision_recall_fscore_support(g.truth,g.prediction,labels=['military','civilian'],average='macro',zero_division=0)
  metrics.append({'model':model,'n':len(g),'accuracy':accuracy_score(g.truth,g.prediction),'precision_macro':p,'recall_macro':r,'f1_macro':f,'coverage':g.prediction.isin(['military','civilian']).mean(),'hard_accuracy':(g[g.hard==1].truth==g[g.hard==1].prediction).mean(),'mean_latency_s':g.latency_s.mean(),'median_latency_s':g.latency_s.median()})
  cm=confusion_matrix(g.truth,g.prediction,labels=['military','civilian','invalid'])[:2,:]
  ax.imshow(cm,cmap='Blues');ax.set_xticks(range(3),['military','civilian','invalid']);ax.set_yticks(range(2),['military','civilian']);ax.set_xlabel('Prediction');ax.set_ylabel('Truth');ax.set_title(model.split('/')[-1])
  for y in range(2):
   for x in range(3):ax.text(x,y,str(cm[y,x]),ha='center',va='center')
 out=ROOT/'results';fig.tight_layout();fig.savefig(out/'confusion_matrices.png',dpi=160);plt.close(fig)
 table=pd.DataFrame(metrics).sort_values('f1_macro',ascending=False);table.to_csv(out/'metrics.csv',index=False)
 errors=df[df.truth!=df.prediction].merge(pd.DataFrame(rows)[['id','page_title','reason','file_url']],on='id');errors.to_csv(out/'errors.csv',index=False)
 a,b=sorted(expected);pv=df.pivot(index='id',columns='model',values='prediction');truth=df[df.model==a].set_index('id').truth
 ca=pv[a].eq(truth);cb=pv[b].eq(truth);n10=int((ca & ~cb).sum());n01=int((~ca & cb).sum());n=n10+n01
 from math import comb
 pvalue=min(1.,2*sum(comb(n,k) for k in range(min(n10,n01)+1))/2**n) if n else 1.
 summary={'status':'complete','n_images':30,'n_responses':60,'a':a,'b':b,'a_only_correct':n10,'b_only_correct':n01,'mcnemar_exact_p':pvalue}
 (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 print(table.to_string(index=False));return table,errors,summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','report']);p.add_argument('--limit',type=int);a=p.parse_args()
 if a.command=='prepare':prepare()
 elif a.command=='run':run(a.limit)
 else:report()
