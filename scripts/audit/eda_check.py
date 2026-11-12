import pandas as pd, re, numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pathlib; P=str(pathlib.Path(__file__).resolve().parents[2]/"data"/"raw")+"/"
def a(t):
    t=t.lower(); t=re.sub(r"\d+","<number>",t); t=re.sub(r"[^a-z<>\s]"," ",t); return re.sub(r"\s+"," ",t).strip()
tr=pd.read_csv(P+"composite_train.csv"); tr["c"]=tr.text.map(a); tr=tr[tr.c.str.len()>0]
df=tr.drop_duplicates(["c","label"]).reset_index(drop=True)
df["wc"]=df.c.str.split().str.len(); df["cc"]=df.c.str.len(); df["uw"]=df.c.str.split().map(lambda x:len(set(x)))
print(df.groupby("label")[["wc","cc","uw"]].agg(["mean","median"]).round(1))
print("pearson",df[["label","wc","cc","uw"]].corr().label.round(3).to_dict())
print("spearman",df[["label","wc","cc","uw"]].corr("spearman").label.round(3).to_dict())
print("wc-only depth-3 tree CV acc",cross_val_score(DecisionTreeClassifier(max_depth=3),df[["wc"]],df.label,cv=5).mean().round(4))
print("majority baseline",round(df.label.mean(),4))
for w in ["xiao","zhang","li","rupees","yuan","customer service","otp","kyc","upi","aadhaar","paytm","sbi"]:
    m=df.c.str.contains(r"\b"+w+r"\b")
    print(f"{w:18s} scam={m[df.label==1].sum():6d} non={m[df.label==0].sum():6d}")
print("pct <number>-containing", df.groupby("label").c.apply(lambda s:s.str.contains("<number>").mean()).round(3).to_dict())
