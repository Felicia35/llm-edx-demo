import asyncio
import json
import math
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

from backend import precompute
from backend.main import DATA_PLACEHOLDER, SYSTEM_PROMPTS, app, render_page


def fake_api():
    tokens = [NS(token=' quiet', bytes=list(b' quiet'), logprob=math.log(0.4)),
              NS(token=' dark', bytes=list(b' dark'), logprob=math.log(0.2)),
              NS(token='\\xe2', bytes=[226], logprob=math.log(0.1))]
    first = NS(token=' quiet', bytes=list(b' quiet'), logprob=math.log(0.4), top_logprobs=tokens)
    result = NS(model='test-model', choices=[NS(message=NS(refusal=None), logprobs=NS(content=[first]))])
    create = AsyncMock(return_value=result)
    app.state.openai = NS(chat=NS(completions=NS(create=create)))
    return create


def test_tree_branches_on_complete_tokens_only():
    create = fake_api()
    data = asyncio.run(precompute.precompute(phrases=['The night was'], depth=3))
    root = data['phrases']['The night was']
    assert list(root['children']) == [' quiet', ' dark']  # Partial UTF-8 token is not a branch.
    assert list(root['children'][' quiet']['children']) == [' quiet', ' dark']
    assert root['children'][' quiet']['children'][' dark']['children'] == {}
    assert precompute.count_nodes(root) == 7
    assert create.call_count == 7 * len(SYSTEM_PROMPTS)
    prompts = {c.kwargs['messages'][1]['content'] for c in create.call_args_list}
    assert 'The night was quiet dark' in prompts
    assert all(c.kwargs['max_completion_tokens'] == 1 and c.kwargs['temperature'] == 1.0 for c in create.call_args_list)
    novels = root['styles']['novels']
    assert novels['tokens'][2]['text'] is None
    assert novels['other_probability'] == 0.3
    assert data['model'] == 'test-model'


def test_render_page_embeds_data_safely():
    data = {'model': '</script><b>', 'phrases': {'__proto__': {}}}
    html = render_page(data)
    assert DATA_PLACEHOLDER not in html
    assert 'const LIVE = false;' in html
    line = next(l for l in html.splitlines() if l.startswith('const DATA = JSON.parse('))
    assert '</script>' not in line
    literal = line[len('const DATA = JSON.parse('):-2].replace('<\\/', '</')
    assert json.loads(json.loads(literal)) == data
