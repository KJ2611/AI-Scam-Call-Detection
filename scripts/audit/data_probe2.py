import pandas as pd, re, numpy as np
import pathlib; P=str(pathlib.Path(__file__).resolve().parents[2]/"data"/"raw")+"/"
tr=pd.read_csv(P+"composite_train.csv")
pd.set_option("display.max_colwidth",220)
def fmt(t):
    t=t.strip()
    if t.startswith('"') or t.startswith("'"): 
        return "list-of-utterances" if re.search(r"['\"]\s*,\s*['\"]",t[:4000]) else "quoted"
    if re.match(r"^(caller|agent|customer|speaker|receiver|innocent|suspect|scammer)\s*[:\d]",t,re.I): return "speaker-tagged"
    if len(t.split())<60: return "short(<60w)"
    return "long-plain"
tr["fmt"]=tr.text.map(fmt)
print(pd.crosstab(tr.fmt,tr.label))
for f in tr.fmt.unique():
    for l in [0,1]:
        s=tr[(tr.fmt==f)&(tr.label==l)]
        if len(s): print("\n##",f,l); print(s.text.sample(min(2,len(s)),random_state=0).str[:300].to_string())
