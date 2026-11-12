import pandas as pd, re, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
import pathlib; P=str(pathlib.Path(__file__).resolve().parents[2]/"data"/"raw")+"/"
tr=pd.read_csv(P+"composite_train.csv")
def a(t):
    t=t.lower(); t=re.sub(r"\d+","<number>",t); t=re.sub(r"[^a-z<>\s]"," ",t); return re.sub(r"\s+"," ",t).strip()
tr["cleaned_text"]=tr.text.map(a); tr=tr[tr.cleaned_text.str.len()>0]
df=tr.drop_duplicates(["cleaned_text","label"]).reset_index(drop=True)
print(len(df), df.label.value_counts().to_dict())
conf=df[df.cleaned_text.duplicated(keep=False)]; print("conflicting-label texts:\n",conf.sort_values("cleaned_text")[["cleaned_text","label"]].to_string()[:800])
X_tr,X_te,y_tr,y_te=train_test_split(df.cleaned_text,df.label,test_size=.2,random_state=42,stratify=df.label)
v=TfidfVectorizer(max_features=10000,ngram_range=(1,2)); A=v.fit_transform(X_tr); B=v.transform(X_te)
for n,m in [("LR",LogisticRegression(max_iter=1000)),("NB",MultinomialNB()),("SVM",LinearSVC())]:
    m.fit(A,y_tr); p=m.predict(B); print(n, round(accuracy_score(y_te,p),4), round(f1_score(y_te,p),4), confusion_matrix(y_te,p).ravel())
# near-duplicate check: test docs whose cleaned text has a train doc with cosine>0.9
from sklearn.neighbors import NearestNeighbors
nn=NearestNeighbors(n_neighbors=1,metric="cosine").fit(A); d,_=nn.kneighbors(B)
sim=1-d.ravel()
for t in [0.8,0.9,0.95,0.99]: print(f"test docs with train neighbour cos>={t}: {(sim>=t).mean():.3f}")
np.save("/tmp/nn_sim.npy",sim)
