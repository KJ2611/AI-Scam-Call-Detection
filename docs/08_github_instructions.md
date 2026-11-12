# Putting this into your existing GitHub repository

The repository URL was not provided, so this project was built as a standalone folder. These commands add it to **your existing repo on a new branch**, without touching `main` and without force-pushing.

```bash
# 1. clone your repo (skip if you already have it locally)
git clone https://github.com/<you>/<your-repo>.git
cd <your-repo>
git checkout main && git pull

# 2. new branch for this work
git checkout -b feature/scam-risk-analyzer-app

# 3. copy the project in (adjust the path to where you unzipped it).
#    rsync keeps your existing files; --ignore-existing never overwrites anything already in your repo.
rsync -av --ignore-existing --exclude '.git' --exclude 'data/raw/*' --exclude 'data/processed/*' \
      /path/to/scam-risk-analyzer/ ./

#    If your repo already has notebooks/ or a README.md, compare before replacing:
#    diff -r /path/to/scam-risk-analyzer/notebooks ./notebooks

# 4. check, commit, push the branch
git status
git add -A
git commit -m "Add Scam Conversation Risk Analyzer app, ML pipeline, docs and tests"
git push -u origin feature/scam-risk-analyzer-app

# 5. on GitHub: open a Pull Request from feature/scam-risk-analyzer-app into main and review it.
```

Notes
- `data/raw/` and `data/processed/` are git-ignored (size and licensing). Everyone runs `data/README.md` setup locally.
- Trained models (`models/*.joblib`, ~4 MB) and reports (`reports/*.json`) are committed so the app runs straight after `pip install -r requirements.txt`.
- The zip you received also contains a local `.git` with one commit on `feature/scam-risk-analyzer-app`. It has no shared history with your repo, so use the rsync route above rather than pushing that `.git` directly.
- If you send me the repository URL, I can do this on a branch for you.
