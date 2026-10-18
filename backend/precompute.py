"""Precompute next-token trees for the fixed phrases on the Open edX page (no API calls in class).

Run once with OPENAI_API_KEY set in backend/.env:  python -m backend.precompute
Writes dist/next-token-data.json (served live at /edx) and dist/next-token-demo.html
(fixed phrases only; can be uploaded to Studio). After editing frontend/edx.html,
`python -m backend.precompute --render` rebuilds the HTML without API requests.
"""
import asyncio
import json
import os
import sys

from fastapi import HTTPException
from openai import AsyncOpenAI

from backend.main import EDX_DATA, MODEL, SYSTEM_PROMPTS, alternatives, app, render_page, request_completion

PHRASES = ['The discovery', 'After the announcement', 'In the beginning']
BRANCH = 5  # Clickable tokens per style; children are the union across styles.
DEPTH = 3   # Distributions along each path, including the starting phrase.
OUT = EDX_DATA.parent


def appendable(token):
    # Only complete UTF-8 tokens can be appended; never introduce a replacement character.
    if token.bytes is None:
        return token.token
    try:
        return bytes(token.bytes).decode('utf-8')
    except UnicodeDecodeError:
        return None


async def distribution(prompt, corpus, limit, models):
    async with limit:
        result = await request_completion(
            [{'role': 'system', 'content': SYSTEM_PROMPTS[corpus]},
             {'role': 'user', 'content': prompt}], 1, 1.0, 10)
    models.add(result.model)
    tokens = alternatives(result.choices[0].logprobs.content[0])
    return {'tokens': [{'token': t.token, 'probability': round(t.probability, 6), 'text': appendable(t)} for t in tokens],
            'other_probability': round(max(0.0, 1.0 - sum(t.probability for t in tokens)), 6)}


async def build(prompt, depth, limit, models):
    results = await asyncio.gather(*(distribution(prompt, c, limit, models) for c in SYSTEM_PROMPTS))
    node = {'styles': dict(zip(SYSTEM_PROMPTS, results)), 'children': {}}
    if depth > 1:
        texts = list(dict.fromkeys(t['text'] for s in results for t in s['tokens'][:BRANCH] if t['text']))
        children = await asyncio.gather(*(build(prompt + text, depth - 1, limit, models) for text in texts))
        node['children'] = dict(zip(texts, children))
    return node


async def precompute(phrases=PHRASES, depth=DEPTH, concurrency=8):
    models = set()
    limit = asyncio.Semaphore(concurrency)
    trees = await asyncio.gather(*(build(p, depth, limit, models) for p in phrases))
    return {'model': ', '.join(sorted(models)), 'temperature': 1.0, 'branch': BRANCH, 'depth': depth,
            'phrases': dict(zip(phrases, trees))}


def count_nodes(node):
    return 1 + sum(count_nodes(child) for child in node['children'].values())


def write_static_page(data):
    (OUT / 'next-token-demo.html').write_text(render_page(data), encoding='utf-8')


async def main():
    if '--render' in sys.argv:
        write_static_page(json.loads(EDX_DATA.read_text(encoding='utf-8')))
        return print(f'Rebuilt {OUT / "next-token-demo.html"} from {EDX_DATA}.')
    key = os.getenv('OPENAI_API_KEY', '').strip()
    if not key:
        raise SystemExit('Set OPENAI_API_KEY in backend/.env first.')
    app.state.openai = AsyncOpenAI(api_key=key, timeout=30.0, max_retries=5)
    worst = len(PHRASES) * len(SYSTEM_PROMPTS) * sum((BRANCH * len(SYSTEM_PROMPTS)) ** d for d in range(DEPTH))
    print(f'Model {MODEL}: up to {worst} one-token requests (fewer when styles share tokens)…')
    try:
        data = await precompute()
    except HTTPException as error:
        raise SystemExit(f'Stopped: {error.detail}') from None
    finally:
        await app.state.openai.close()
    OUT.mkdir(exist_ok=True)
    EDX_DATA.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
    write_static_page(data)
    nodes = sum(count_nodes(tree) for tree in data['phrases'].values())
    print(f'Done: {nodes * len(SYSTEM_PROMPTS)} requests, {nodes} texts. Upload {OUT / "next-token-demo.html"} to Studio.')


if __name__ == '__main__':
    asyncio.run(main())
