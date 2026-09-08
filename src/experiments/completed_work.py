"""Continue fixed, verified generated work through controlled competing work."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import fcntl
import json
from pathlib import Path
import random
import re

import yaml

from reasoning_trajectory.artifacts import read_generation_rows
from src.experiments.dependency_state import parse_result
from src.experiments.state_routing_heads import _render_trace, parse_generated_writes
from src.models.generation_pipeline import generate_task
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.runtime.artifact_store import write_json


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def generations(root):
    return read_generation_rows(root) if (root / 'generation/generations.jsonl').exists() else []


def verified_prefix(case, generation):
    text = generation['generation']['text']
    names = [x['name'] for x in case['clean']['writes']]
    writes = parse_generated_writes(text, names)
    for event in range(3):
        start, end = case['clean']['spans'][f'write_{event}']
        expected = int(case['clean']['text'][start:end])
        if not any(w['event'] == event and w['literal_digit'] and w['raw_value'] == expected for w in writes):
            return None
    source = next((w for w in writes if w['event'] == 2 and w['literal_digit']), None)
    if source is None or source['raw_value'] != case['clean_answer']:
        return None
    start, end = source['char_span']
    left = text.rfind('\n', 0, start) + 1
    right = text.find('\n', end)
    right = len(text) if right < 0 else right
    if not re.search(r'(?:=|→|->)\s*$', text[left:start]):
        return None
    if not re.fullmatch(r'\s*[.\s]*(?:\(mod\s*10\))?[.\s]*', text[end:right], re.I):
        return None
    lines = text[:right].splitlines()
    numbered = [line for line in lines if re.match(r'^\s*[123][.)]\s', line)]
    return '\n'.join(numbered) + '\n' if len(numbered) == 3 else None


def source_cases(source, model, count):
    folder = source / 'runs' / model / 'design_2026_09_07/R04_state_routing_heads'
    cases = {r['id']: r for r in read_rows(folder / 'dataset.jsonl')}
    eligible = []
    for generation in read_rows(folder / 'evaluation/measurements.jsonl'):
        if generation['split'] != 'test':
            continue
        case = cases[generation['id']]
        prefix = verified_prefix(case, generation)
        if prefix is not None:
            eligible.append((case, prefix))
    eligible.sort(key=lambda pair: hashlib.sha256(pair[0]['id'].encode()).hexdigest())
    # Eligibility uses only the first three writes, never the original answer.
    if len(eligible) < count:
        raise ValueError(f'{model}: only {len(eligible)} verified prefixes, need {count}')
    return eligible[:count], len(eligible)


def variants(case, prefix, seed):
    initial = {name: int(value) for name, value in re.findall(r'(Ada|Bela|Cora|Dani) has (\d)', case['question'].split('Events, in order:')[0])}
    events = [{'name': name, 'operation': op, 'amount': int(amount)} for name, op, amount in re.findall(r'\d+\. (Ada|Bela|Cora|Dani) must (add|subtract) (\d)\.', case['question'])][:3]
    if len(initial) != 4 or len(events) != 3:
        raise ValueError('source question contract changed')
    rng = random.Random(seed + int(case['id'].split('_')[-1]))
    others = [name for name in initial if name != case['target']]
    tail = [{'name': rng.choice(others), 'operation': rng.choice(['add', 'subtract']), 'amount': rng.randint(1, 8)} for _ in range(64)]
    result = []
    for horizon in (0, 8, 32, 64):
        teacher, gold = _render_trace(initial=initial, events=events + tail[:horizon], target=case['target'])
        assert gold == case['clean_answer']
        question, working = teacher['text'].split('Solution:\n')
        question = question.replace('Use one short line per event, then end with Answer=<one digit>.',
            'Finish any remaining work. End with exactly Answer: [one digit].')
        history = '\n'.join(working.splitlines()[3:-2]) + '\n' if horizon else ''
        common = {'source_id': case['id'], 'target': case['target'], 'horizon': horizon,
                  'expected': [gold], 'question': question, 'own_prefix': prefix,
                  'oracle_history': history}
        for mode in ('native_compute', 'oracle_history', 'matched_filler'):
            if horizon == 0 and mode != 'oracle_history':
                continue
            result.append(dict(common, experiment='R1_intervening_work', arm=mode,
                               prefix_mode=mode, reminder='none'))
        if horizon == 32:
            for order in ('target_early', 'target_late'):
                result.append(dict(common, experiment='R2_independent_order', arm=order,
                                   prefix_mode=order, reminder='none',
                                   question=question.replace('Work through all events in order.', 'Events for different players may be evaluated in either order.')))
            for reminder in ('neutral', 'target', 'correct_value', 'wrong_value'):
                result.append(dict(common, experiment='R3_target_selection', arm=reminder,
                                   prefix_mode='oracle_history', reminder=reminder))
    return result


def prepare(config, source, out):
    cells = []
    counts = {}
    for model in config['models']:
        sources, available = source_cases(source, model, config['prefixes_per_model'][model])
        counts[model] = {'eligible': available, 'selected': len(sources), 'screened': 480}
        for shard in range(config['shards']):
            rows = []
            for index, (case, prefix) in enumerate(sources):
                if index % config['shards'] != shard:
                    continue
                for row in variants(case, prefix, config['seed']):
                    row['id'] = f"{case['id']}:{row['experiment']}:{row['horizon']}:{row['arm']}"
                    rows.append(row)
            cells.append({'model': model, 'shard': shard, 'rows': rows})
    lock = {'config': config, 'counts': counts, 'cells': cells,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = out / 'lock.json'
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.prepare.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if path.exists() and json.loads(path.read_text()) != lock:
            raise ValueError('locked design changed')
        if not path.exists():
            write_json(path, lock)
    return lock


def prefill(tokenizer, row):
    own, history = row['own_prefix'], row['oracle_history']
    mode = row['prefix_mode']
    if mode == 'native_compute':
        return own
    if mode == 'target_late':
        first, independent, last = own.splitlines()
        text = independent + '\n' + history + first + '\n' + last + '\n'
    elif mode == 'matched_filler':
        budget = len(tokenizer.encode(history, add_special_tokens=False))
        # Pad with a fixed non-numeric token. Matching happens separately for
        # each tokenizer, and the realized token counts are saved.
        token = tokenizer.encode(' note', add_special_tokens=False)[-1]
        target_count = len(tokenizer.encode(own + history + 'Answer: ', add_special_tokens=False))
        for _ in range(4):
            text = own + tokenizer.decode([token] * budget) + '\n'
            difference = target_count - len(tokenizer.encode(text + 'Answer: ', add_special_tokens=False))
            if difference == 0:
                break
            budget += difference
        else:
            raise ValueError('filler does not match the complete prefix length')
    else:
        text = own + history
    if mode in ('target_early', 'target_late'):
        text = '\n'.join(f"{i + 1}. " + re.sub(r'^\d+[.)]\s*', '', line) for i, line in enumerate(text.splitlines())) + '\n'
    reminder = row['reminder']
    if reminder != 'none':
        value = row['expected'][0]
        cues = {'target': f"The requested player is {row['target']}.\n",
            'correct_value': f"The requested player is {row['target']}; the earlier result is {value}.\n",
            'wrong_value': f"The requested player is {row['target']}; the earlier result is {(value + 1) % 10}.\n",
            'neutral': 'The requested answer is one final digit.\n'}
        target_count = max(len(tokenizer.encode(text + c + 'Answer: ', add_special_tokens=False)) for c in cues.values())
        padding = ''
        for _ in range(20):
            current = text + padding + cues[reminder] + 'Answer: '
            count = len(tokenizer.encode(current, add_special_tokens=False))
            if count == target_count:
                return current
            if count > target_count:
                raise ValueError('reminder length cannot be matched')
            padding += ' note'
        raise ValueError('reminder padding failed')
    return text + 'Answer: '


def reduce_cell(root, rows):
    lookup = {r['id']: r for r in rows}
    groups = defaultdict(list)
    measurements = []
    for generation in generations(root):
        row = lookup[generation['sample_id']]
        strict = parse_result(generation['produced_text'])
        match = re.search(r'\bAnswer:\s*(?:\[(\d)\]|(\d))\s*$', generation['produced_text'])
        parsed = [int(match[1] or match[2])] if match else None
        item = {'id': row['id'], 'source_id': row['source_id'], 'experiment': row['experiment'],
                'horizon': row['horizon'], 'arm': row['arm'], 'prediction': parsed,
                'correct': parsed == row['expected'], 'parsed': parsed is not None,
                'strict_list_correct': strict == row['expected'],
                'tokens': len(generation['generated_token_ids'])}
        measurements.append(item)
        groups[(row['experiment'], row['horizon'], row['arm'])].append(item)
    summary = {'expected': len(rows), 'observed': len(measurements), 'measurements': measurements,
        'groups': [{'experiment': k[0], 'horizon': k[1], 'arm': k[2], 'n': len(v),
                    'correct': sum(x['correct'] for x in v), 'parsed': sum(x['parsed'] for x in v)} for k, v in groups.items()]}
    write_json(root / 'evaluation/summary.json', summary)
    return summary


def run(config, source, out, index, smoke=False):
    cell = prepare(config, source, out)['cells'][index]
    rows = cell['rows']
    if smoke:
        ids = list(dict.fromkeys(r['source_id'] for r in rows))[:2]
        rows = [r for r in rows if r['source_id'] in ids and (r['horizon'] in (0, 8) or r['experiment'] != 'R1_intervening_work')]
    root = out / ('smoke' if smoke else 'cells') / cell['model'] / f"shard{cell['shard']}"
    old = yaml.safe_load((source / 'runs' / cell['model'] / 'design_2026_09_07/R12_dependency_revision/config.yaml').read_text())
    cfg = {'model': old['model'], 'generation': {'max_new_tokens': config['max_new_tokens'],
        'num_samples_per_item': 1, 'base_seed': config['seed'], 'temperature': 0.,
        'stop_regex': r'Answer:\s*\[\s*\d\s*\]'},
        'capture': {'enabled': False}, 'prompt': {'mode': 'chat',
        'instruction': 'Use the supplied partial work and finish the requested calculation. End with Answer: [one digit].'}}
    model, tokenizer = load_hf_model_and_tokenizer(cfg['model'])
    done = {r['sample_id'] for r in generations(root)}
    contracts = []
    for index, row in enumerate(rows):
        prefix = prefill(tokenizer, row)
        contracts.append({'id': row['id'], 'prefix_tokens': len(tokenizer.encode(prefix, add_special_tokens=False)),
                          'history_tokens': len(tokenizer.encode(row['oracle_history'], add_special_tokens=False))})
        if row['id'] not in done:
            sample = {'id': row['id'], 'question': row['question'], 'gold_answer': str(row['expected'])}
            budget = config['max_new_tokens'] if row['prefix_mode'] == 'native_compute' else 32
            generated_cfg = dict(cfg, generation=dict(cfg['generation'], forced_prefix=prefix,
                max_new_tokens=len(tokenizer.encode(prefix, add_special_tokens=False)) + budget))
            generate_task(run_path=root, config=generated_cfg, model=model, tokenizer=tokenizer,
                sample=sample, sample_index=index, sample_iter=0, progress=None, progress_label=row['id'])
    write_json(root / 'evaluation/token_contract.json', contracts)
    result = reduce_cell(root, rows)
    if result['observed'] != len(rows):
        raise ValueError('missing generations')
    controls = [r for r in result['measurements'] if r['horizon'] == 0]
    ready = not smoke or (controls and all(r['parsed'] for r in controls) and sum(r['correct'] for r in controls) >= len(controls) * .5)
    write_json(root / 'complete.json', {'complete': True, 'ready': bool(ready), 'smoke': smoke})
    if not ready:
        raise RuntimeError('short control failed; full run remains blocked')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--phase', choices=['prepare', 'smoke', 'run', 'reduce'], required=True)
    p.add_argument('--cell', type=int, default=0)
    a = p.parse_args(); config = yaml.safe_load(a.config.read_text())
    if a.phase == 'prepare':
        print(json.dumps({'cells': len(prepare(config, a.source_root, a.out)['cells'])}))
    elif a.phase in ('smoke', 'run'):
        run(config, a.source_root, a.out, a.cell, a.phase == 'smoke')
    else:
        for cell in prepare(config, a.source_root, a.out)['cells']:
            reduce_cell(a.out / 'cells' / cell['model'] / f"shard{cell['shard']}", cell['rows'])


if __name__ == '__main__':
    main()
