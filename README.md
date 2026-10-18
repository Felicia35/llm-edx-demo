# Open edX next-token demo — backend

One underlying LLM, four system prompts: novels, lyrics, news, academic.
This activity measures how instructions affect the first assistant-output token's
probability distribution. It does **not** train four models or measure the
causal effect of changing pretraining data. A chat continuation is conditioned
on message boundaries and instructions as well as the learner's text.

## Run locally

From this project folder (Python 3.10 or newer):

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
```

Set `OPENAI_API_KEY` in `backend/.env` using your editor, then run:

```sh
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The key stays on the backend; `.env` is ignored by Git. A blank `.env` has been
created for you. Restart after changing it. `OPENAI_MODEL` defaults to
`gpt-4.1-mini`; a replacement must support Chat Completions, logprobs,
temperature, and a one-token output limit.

Interactive API docs: http://localhost:8000/docs

Frontend: http://localhost:8000 — enter text, select one style or compare all four,
adjust temperature, and append a sampled token. The frontend uses the same
server, so no separate frontend process is needed.

## Curl test

```sh
curl --fail-with-body http://localhost:8000/health
curl --fail-with-body http://localhost:8000/next-token \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The night was","corpus":"novels","temperature":1,"top_logprobs":10}'
```

Repeat with `lyrics`, `news`, and `academic`, keeping all other inputs identical.
Each request is a paid API call. No mocked probabilities are used by the endpoint.

The response includes `generated_token`, its logprob/probability/UTF-8 bytes,
`tokens` (each with token, bytes, logprob, probability), `other_probability`,
corpus, resolved model, and requested temperature. Probabilities are
`exp(logprob)` with no renormalization. “Other” is the remaining mass outside
the returned alternatives, subject to API rounding. Missing alternatives are
unknown, not zero probability. Preserve leading whitespace when displaying or
appending tokens. Tokens may be word fragments or partial UTF-8 characters;
combine returned bytes before decoding a sequence of tokens.

The API reports logprobs for assistant output, not raw base-model logits. Keep
temperature at 1 for the initial comparison. Do not reconstruct an exact full
vocabulary temperature distribution from only the top 10 or 20 alternatives.

## Error handling and local browser access

Invalid input returns 422, missing backend credentials 503, upstream
connection/auth/model failures 502, upstream quota/rate limits 429, and
upstream timeouts 504. Provider error details and secrets are not returned.

Local frontend origins on ports 5500 and 8001 are allowed by default. Serve a
future frontend over HTTP rather than opening a `file://` page. Set
`CORS_ORIGINS` to comma-separated exact origins for Open edX integration.
This is a local prototype: before public deployment, add server-side learner
access control and usage limits. CORS alone does not authorize API callers.

## Automated checks (no API key or billed calls)

```sh
python -m pip install -r backend/requirements-dev.txt
python -m pytest -q
```

Tests stub the provider to verify all four prompts, request parameters,
probability conversion, input validation, CORS, and error handling. They do not
establish live model availability; the curl request with your key does that.

## Official API references

- [Chat Completions request and logprobs schema](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini)

## Open edX static page (no backend in class)

Students pick one of three fixed phrases, then click any style's top-5 token
(or sample one) to append it and compare all four styles again, three steps
deep. Every distribution is precomputed once, so the course needs no server,
API key, or per-student cost.

```sh
python -m backend.precompute   # uses OPENAI_API_KEY / OPENAI_MODEL from backend/.env
```

This writes `dist/next-token-demo.html` (data embedded) and
`dist/next-token-data.json` (the same data, for inspection). Phrases, branching,
and depth are constants at the top of `backend/precompute.py`; requests grow as
(5 × 4 styles)^depth in the worst case. The template is `frontend/edx.html`.

In Studio, upload `next-token-demo.html` under **Content → Files**, then add an
HTML component (Raw HTML editor) containing:

```html
<iframe src="/static/next-token-demo.html" title="Next-token explorer"
        style="width:100%;height:1500px;border:0"></iframe>
```

Probabilities are a snapshot from one run at temperature 1, so the page has no
temperature control; rerun the script to refresh them.

## Open edX with students' own text (Google Cloud Run)

Served by the backend at `/edx`, the same page also lets students type their
own text and keep adding tokens beyond the prepared paths; those steps call
`/next-token` live (four requests each). Prepared paths still cost nothing.

Two optional env vars protect the deployed backend:

- `ACCESS_TOKEN`: `/next-token` and `/chat` require a matching `X-Access-Token`
  header. The page reads it from its URL (`/edx?key=…`). It sits in the course
  page source, so it stops casual misuse of the URL, not a determined student.
- `DAILY_REQUEST_LIMIT`: upstream requests allowed per day (0 = unlimited). The
  count lives in memory, so deploy with one instance; it resets when the
  instance restarts. Keep a hard budget limit in your OpenAI project as well.

One-time setup, with the [gcloud CLI](https://cloud.google.com/sdk/docs/install)
and a project that has billing enabled:

```sh
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
grep '^OPENAI_API_KEY=' backend/.env | cut -d= -f2- | tr -d '\n' | gcloud secrets create openai-api-key --data-file=-
python3 -c "import secrets; print(secrets.token_urlsafe(24), end='')" | gcloud secrets create next-token-access --data-file=-
NUMBER=$(gcloud projects describe "$(gcloud config get-value project)" --format='value(projectNumber)')
for s in openai-api-key next-token-access; do
  gcloud secrets add-iam-policy-binding $s --role=roles/secretmanager.secretAccessor \
    --member=serviceAccount:$NUMBER-compute@developer.gserviceaccount.com
done
```

Deploy (and redeploy after changes) from this folder. `.gcloudignore` keeps
`.env` out of the upload; `dist/next-token-data.json` must exist first.

```sh
gcloud run deploy next-token-lab --source . --region northamerica-northeast2 \
  --allow-unauthenticated --max-instances 1 \
  --set-secrets OPENAI_API_KEY=openai-api-key:latest,ACCESS_TOKEN=next-token-access:latest \
  --set-env-vars OPENAI_MODEL=gpt-4.1-mini,DAILY_REQUEST_LIMIT=10000
```

Print the access token with
`gcloud secrets versions access latest --secret next-token-access`, then use this
in a Raw HTML component (no file upload needed):

```html
<iframe src="https://YOUR-SERVICE-URL/edx?key=YOUR_ACCESS_TOKEN" title="Next-token explorer"
        style="width:100%;height:2800px;border:0"></iframe>
```

To change the token, add a new secret version and redeploy; update the iframe.

## Making differences visible

Each style now has original few-shot continuation examples using the same three
prefixes across styles. This encourages suffix-only output and demonstrates
more distinct genre patterns. These examples are part of the conditioning
context, not new training data. The frontend also aligns exact matching tokens
in a comparison table. Tokens omitted by the API are marked unknown, not zero.
Try several open prefixes at temperature 1. Similar results are valid, and
stronger prompting cannot guarantee distinct distributions for every prefix.

## Chat comparison

The chat panel sends identical user/assistant history to `/chat` for each style.
Choose one reply to form the shared history for the next turn. Conversations
stay in browser memory and reset on reload. Each turn makes four API requests.
Replies are limited to 160 output tokens; truncation is labeled.

The response includes full text plus each output position's token, bytes,
logprob, probability, top 10 alternatives, and remaining probability mass.
The first-token table compares identical conversation context across styles.
Later positions condition on each reply's own preceding tokens; they are not
controlled cross-style comparisons. These probabilities provide generation
evidence, not proof of genre, factual correctness, or different pretraining.
