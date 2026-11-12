import pandas as pd, re, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix
import pathlib; P=str(pathlib.Path(__file__).resolve().parents[2]/"data"/"raw")+"/"
def a(t):
    t=t.lower(); t=re.sub(r"\d+","<number>",t); t=re.sub(r"[^a-z<>\s]"," ",t); return re.sub(r"\s+"," ",t).strip()
tr=pd.read_csv(P+"composite_train.csv"); tr["c"]=tr.text.map(a); tr=tr[tr.c.str.len()>0]
df=tr.drop_duplicates(["c","label"]).reset_index(drop=True)
X_tr,X_te,y_tr,y_te,r_tr,r_te=train_test_split(df.c,df.label,df.text,test_size=.2,random_state=42,stratify=df.label)
# 1) shortcut: raw-format features only, no words
def feats(raw):
    return np.c_[raw.str.len(), raw.str.split().str.len(), raw.str.count("', '")+raw.str.count('", "'), raw.str.startswith("'")|raw.str.startswith('"'), raw.str.count(r"\?")]
rf=RandomForestClassifier(200,random_state=42,n_jobs=-1).fit(feats(r_tr),y_tr); p=rf.predict(feats(r_te))
print("FORMAT/LENGTH-ONLY RF  acc",round(accuracy_score(y_te,p),4),"f1",round(f1_score(y_te,p),4))
# 2) main SVM
v=TfidfVectorizer(max_features=10000,ngram_range=(1,2)); A=v.fit_transform(X_tr); B=v.transform(X_te)
svm=LinearSVC().fit(A,y_tr); lr=LogisticRegression(max_iter=1000).fit(A,y_tr)
sim=np.load("/tmp/nn_sim.npy")
mask=sim<0.9
for n,m in [("SVM",svm),("LR",lr)]:
    p=m.predict(B); print(n,"full test acc",round(accuracy_score(y_te,p),4)," | test w/o near-dups(cos<0.9) n=",mask.sum(),"acc",round(accuracy_score(y_te[mask],p[mask]),4),"f1",round(f1_score(y_te[mask],p[mask]),4))
# 3) external datasets
s=open(P+"English_Scam.txt",encoding="utf-8",errors="replace").read(); n=open(P+"English_NonScam.txt",encoding="utf-8",errors="replace").read()
sp=[re.sub(r"^\d+\.\s*","",x.strip()) for x in re.split(r"\n\s*\n",s) if x.strip()]; npp=[x.strip() for x in re.split(r"\n\s*\n",n) if x.strip()]
eng=pd.DataFrame({"text":sp+npp,"label":[1]*len(sp)+[0]*len(npp)})
fc=pd.read_csv(P+"fraud_call.file",sep="\t",header=None,names=["label","text"],quoting=3,on_bad_lines="skip"); fc["label"]=(fc.label=="fraud").astype(int); fc=fc.dropna().drop_duplicates("text")
ct=pd.read_csv(P+"composite_test.csv"); ct["c"]=ct.text.map(a); ct=ct[ct.c.str.len()>0].drop_duplicates(["c","label"])
trainset=set(df.c)
ct_new=ct[~ct.c.isin(trainset)]
print("composite_test unique",len(ct),"overlap w/ composite_train (cleaned exact)",ct.c.isin(trainset).sum())
fc["c"]=fc.text.map(a); print("fraud_call rows",len(fc),"also in composite_train:",fc.c.isin(trainset).sum())
eng["c"]=eng.text.map(a); print("English rows",len(eng),"also in composite_train:",eng.c.isin(trainset).sum())
for name,d in [("composite_test (unseen rows only)",ct_new),("English_Scam/NonScam",eng),("fraud_call.file",fc[~fc.c.isin(trainset)])]:
    Bx=v.transform(d.c)
    for mn,m in [("SVM",svm),("LR",lr)]:
        p=m.predict(Bx); y=d.label.values
        print(f"{name:35s} {mn} n={len(d)} acc={accuracy_score(y,p):.4f} P={precision_score(y,p,zero_division=0):.4f} R={recall_score(y,p):.4f} F1={f1_score(y,p):.4f} cm={confusion_matrix(y,p).ravel()}")
