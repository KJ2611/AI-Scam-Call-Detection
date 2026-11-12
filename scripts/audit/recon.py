import pandas as pd, re
import pathlib; P=str(pathlib.Path(__file__).resolve().parents[2]/"data"/"raw")+"/"
tr=pd.read_csv(P+"composite_train.csv")
def variants():
    def a(t):
        t=t.lower(); t=re.sub(r"\d+","<number>",t); t=re.sub(r"[^a-z<>\s]"," ",t); return re.sub(r"\s+"," ",t).strip()
    def b(t):
        t=t.lower(); t=re.sub(r"http\S+|www\.\S+"," ",t); t=re.sub(r"\d+(?:[.,]\d+)*"," <number> ",t); t=re.sub(r"[^a-z<>\s]","",t); return re.sub(r"\s+"," ",t).strip()
    def c(t):
        t=t.lower(); t=re.sub(r"\d+","<number>",t); t=re.sub(r"[^\w\s<>]","",t); return re.sub(r"\s+"," ",t).strip()
    def d(t):
        t=t.lower(); t=re.sub(r"http\S+|www\.\S+","",t); t=re.sub(r"\d+(?:\.\d+)?","<number>",t); t=re.sub(r"[^a-z0-9<>\s]","",t); return re.sub(r"\s+"," ",t).strip()
    return dict(a=a,b=b,c=c,d=d)
for k,f in variants().items():
    x=tr.copy(); x["ct"]=x.text.map(f); x=x[x.ct.str.len()>0]
    d1=x.drop_duplicates("ct"); d2=x.drop_duplicates(["ct","label"]); d3=x.drop_duplicates("text")
    print(k, "dedup ct:",len(d1),d1.label.value_counts().to_dict()," dedup ct+label:",len(d2)," dedup raw:",len(d3), d3.label.value_counts().to_dict())
x=tr.drop_duplicates("text"); print("raw dedup then",len(x))
