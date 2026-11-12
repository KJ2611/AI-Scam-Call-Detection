import pandas as pd, re, sys
import pathlib; P=str(pathlib.Path(__file__).resolve().parents[2]/"data"/"raw")+"/"
tr=pd.read_csv(P+"composite_train.csv"); te=pd.read_csv(P+"composite_test.csv")
print("composite_train",tr.shape, tr.label.value_counts().to_dict()); print("composite_test",te.shape, te.label.value_counts().to_dict())
print("train dup texts", tr.text.duplicated().sum(), " test dup", te.text.duplicated().sum())
print("exact overlap train/test", len(set(tr.text)&set(te.text)))
print(tr.label.unique(), tr.text.isna().sum())
fc=pd.read_csv(P+"fraud_call.file",sep="\t",header=None,names=["label","text"],quoting=3,on_bad_lines="warn")
print("fraud_call",fc.shape,fc.label.value_counts().to_dict(),"dups",fc.text.duplicated().sum())
s=open(P+"English_Scam.txt",encoding="utf-8",errors="replace").read(); n=open(P+"English_NonScam.txt",encoding="utf-8",errors="replace").read()
sp=[x.strip() for x in re.split(r"\n\s*\n",s) if x.strip()]; npp=[x.strip() for x in re.split(r"\n\s*\n",n) if x.strip()]
print("English_Scam paras",len(sp),"NonScam paras",len(npp))
print("unique scam",len(set(sp)),"unique non",len(set(npp)))
# length stats
for name,d in [("tr",tr),("te",te)]:
    wc=d.text.str.split().str.len(); print(name, wc.groupby(d.label).describe().round(1).to_string())
print(tr.sample(5,random_state=1).to_string()[:2500])
