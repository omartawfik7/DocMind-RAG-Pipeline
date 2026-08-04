# DocMind Deployment Guide — $0/month

This turns DocMind from "runs on my laptop" into a real hosted system: a live
URL, a persistent vector store that survives restarts, and secrets that
aren't sitting in your git history.

## What changed from your original code, and why

| Change | Why |
|---|---|
| Groq key moved from hardcoded string to `GROQ_API_KEY` env var | The old key was exposed in your public repo. Never commit secrets. |
| ChromaDB (local) → Qdrant Cloud (hosted) | Render's free web-service tier has no persistent disk — local files don't reliably survive a restart or redeploy. Qdrant Cloud's free tier is a real external database, so your ingested documents persist independently of the app server. |
| `/api/health` now reports `llama-3.3-70b-versatile` instead of the old hardcoded `gpt-4o` | The health check was reporting a model you never call. Small bug, worth fixing since it's the first thing anyone checking your API sees. |
| Added `templates/index.html` | This file was referenced by `app.py` and by your own README but wasn't actually in the GitHub repo — the app would 404/500 on `/` without it. Built a minimal working chat UI so deployment isn't blocked; restyle it however you like. |
| Added `gunicorn` + `Procfile` | Flask's built-in dev server (`app.run()`) isn't meant for anything other than local development. Gunicorn is the standard production WSGI server. |
| Added `.env.example` + `.gitignore` | So the next secret you add doesn't end up in git by accident. |
| Added `eval/` folder | A reproducible retrieval-quality evaluation harness — the thing that turns "I built a RAG pipeline" into "I built a RAG pipeline and I can show you its measured retrieval accuracy." |

---

## Step 0 — Rotate your Groq key (if you haven't already)

Go to **console.groq.com/keys**, delete the old exposed key, and create a new one.
Keep it somewhere private for now — you'll paste it into Render's environment
settings in Step 3, never into code.

## Step 1 — Create a free Qdrant Cloud cluster

1. Go to **cloud.qdrant.io** and sign up (no credit card).
2. Create a cluster — pick the **Free** tier (0.5 vCPU / 1GB RAM / 4GB disk, permanent, $0).
3. Once it's provisioned, copy two things from the cluster dashboard:
   - The **cluster URL** (looks like `https://xxxxx.us-east.aws.cloud.qdrant.io`)
   - An **API key** (create one under the cluster's API Keys tab)

**One thing to know:** free Qdrant clusters suspend after 1 week of no activity and delete after 4 weeks if never reactivated. For an active portfolio piece this is a non-issue — normal traffic (you demoing it, a recruiter clicking the link) keeps it alive. If it goes quiet for a month, you'll just need to log in and reactivate the cluster before your next interview.

## Step 2 — Push the updated code to your repo

Replace `app.py`, `rag_engine.py`, `requirements.txt` in your GitHub repo with
the versions here, and add the new files: `Procfile`, `render.yaml`,
`.env.example`, `.gitignore`, `templates/index.html`, and the `eval/` folder.

```bash
git add app.py rag_engine.py requirements.txt Procfile render.yaml \
        .env.example .gitignore templates/ eval/
git commit -m "Deploy: move secrets to env vars, switch to Qdrant Cloud, add eval harness"
git push
```

**Double-check `.env` itself is never committed** — `.gitignore` now excludes it, but if you ever created a local `.env` before this, make sure it isn't already tracked (`git rm --cached .env` if it is).

## Step 3 — Deploy on Render

1. Go to **render.com**, sign up (no card needed for the free path), and connect your GitHub account.
2. **New → Blueprint**, point it at your `DocMind-RAG-Pipeline` repo. Render will read `render.yaml` automatically and set up the service.
   - If you'd rather do it manually instead of via the blueprint: **New → Web Service**, connect the repo, set Build Command to `pip install -r requirements.txt` and Start Command to `gunicorn app:app --timeout 120`.
3. In the service's **Environment** tab, add:
   - `GROQ_API_KEY` = your new Groq key
   - `QDRANT_URL` = your Qdrant cluster URL
   - `QDRANT_API_KEY` = your Qdrant API key
4. Deploy. First build will take a few minutes (installing `sentence-transformers` pulls in PyTorch, which is sizeable — this is normal).
5. Render gives you a live URL like `https://docmind-rag-pipeline.onrender.com`. That's the link for your resume/portfolio.

**Cold starts:** the free tier spins down after 15 minutes of inactivity and takes ~30-60 seconds to wake back up on the next request. Worth mentioning up front if you're sending someone a live link ("first load may take a minute to spin up — free tier").

## Step 4 — Run the retrieval eval

Once your env vars work locally (or against the deployed instance, by pointing your local `.env` at the same Qdrant cluster), run:

```bash
cd eval
python eval_retrieval.py
```

This ingests a reference document with 8 known facts, asks 10 questions against it, and reports:
- **Hit-rate@6** — what fraction of questions retrieved a chunk containing the right answer, in the top 6 results
- **MRR** — how highly ranked the correct chunk was, on average
- **Average retrieval latency**

It writes `eval_report.json` with the full breakdown per question. This is the artifact from the earlier conversation — something concrete to point to that shows you measure your system's retrieval quality rather than just eyeballing whether answers look reasonable.

## What this gets you for the resume/interview

- A **live URL** you can put next to the DocMind bullet, instead of just a repo link.
- A real answer to "how do you know it works" — the eval report, with numbers.
- A real answer to "how did you handle secrets/deployment" — env vars, no hardcoded keys, a WSGI server instead of Flask's dev server, a persistence layer that survives the app restarting.
- A concrete, honest story about the free-tier/persistent-disk tradeoff — that's a genuine systems-design decision, and being able to explain *why* you picked Qdrant Cloud over local storage is a better interview answer than the deployment itself.
