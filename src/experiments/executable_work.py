"""Test selection and use of completed work in executable integer programs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import random

import yaml

from reasoning_trajectory.artifacts import read_generation_rows
from src.experiments.dependency_state import program_facts, parse_result
from src.models.generation_pipeline import generate_task
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.runtime.artifact_store import write_json


def cases(config):
    rng = random.Random(config['seed'])
    rows = []
    for index in range(config['problems']):
        # Keep arithmetic easy while increasing independent work. Unlike the
        # routing task, values are ordinary integers with no modulo operation.
        branches = []
        for branch in range(33):
            initial = rng.randint(20, 39)
            first, second = rng.randint(2, 9), rng.randint(2, 9)
            name = f'v{branch:02d}'
            branches.append([f'{name}_0 = {initial}', f'{name}_1 = {name}_0 + {first}',
                             f'{name}_2 = {name}_1 + {second}'])
        for width in config['distractors']:
            lines = [line for branch in branches[:width + 1] for line in branch]
            for consumer in config.get('consumers', ('lookup', 'combine')):
                expression = {'lookup': '(v00_2,)', 'copy': '(v00_2, v01_2, 0)',
                              'combine': '(v00_2, v01_2, v00_2 + v01_2)'}[consumer]
                code = '\n'.join(lines + ['result = ' + expression])
                facts = program_facts(code)['values']
                trace = [f"{line.split(' = ')[0]} = {facts[line.split(' = ')[0]]}" for line in lines]
                base = {'problem': index, 'horizon': width, 'consumer': consumer,
                        'code': code, 'expected': list(facts['result']),
                        'target_value': facts['v00_2'], 'other_required': facts['v01_2'],
                        'last_value': facts[f'v{width:02d}_2'],
                        'last_agrees': facts[f'v{width:02d}_2'] == facts['v00_2'],
                        'question': 'Compute result for this Python program. Independent branches may be evaluated in either order.\n```python\n' + code + '\n```'}
                for arm in config.get('arms', ('native', 'early', 'late', 'filler', 'name_cue', 'last_value_hidden', 'last_name_hidden')):
                    work = list(trace)
                    if arm in ('late', 'forced_late'):
                        work = trace[3:] + trace[:3]
                    if arm == 'both_late':
                        work = trace[6:] + trace[:6]
                    if arm == 'arithmetic_only':
                        work = [trace[2], trace[5]]
                    if arm == 'last_value_hidden':
                        work[-1] = work[-1].split(' = ')[0] + ' = [omitted]'
                    if arm == 'last_name_hidden':
                        work[-1] = '[omitted] = ' + str(base['last_value'])
                    rows.append(dict(base, arm=arm, work='\n'.join(work) + '\n',
                        # Preserve both required prerequisites in the filler
                        # condition. Only irrelevant records become padding.
                        kept='\n'.join(trace[:6]) + '\n',
                        irrelevant='\n'.join(trace[6:]) + '\n',
                        id=f"program-{index:04d}:w{width}:{consumer}:{arm}"))
    return rows


def prepare(config, out):
    rows = cases(config)
    cells = [{'model': model, 'shard': shard, 'rows': [r for r in rows if r['problem'] % config['shards'] == shard]}
             for model in config['models'] for shard in range(config['shards'])]
    lock = {'config': config, 'cells': cells,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.prepare.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        path = out / 'lock.json'
        if path.exists() and json.loads(path.read_text()) != lock:
            raise ValueError('registered input changed')
        if not path.exists():
            write_json(path, lock)
    return lock


def matched_padding(tokenizer, head, tail, target):
    token = tokenizer.encode(' note', add_special_tokens=False)[-1]
    budget = target - len(tokenizer.encode(head + tail, add_special_tokens=False))
    for _ in range(8):
        text = head + tokenizer.decode([token] * max(0, budget)) + tail
        difference = target - len(tokenizer.encode(text, add_special_tokens=False))
        if difference == 0:
            return text
        budget += difference
    raise ValueError('token-matched padding failed')


def prefill(tokenizer, row):
    if row['arm'] == 'native':
        return ''
    text = row['work']
    ending = f"Return exactly {len(row['expected'])} integers.\nAnswer: ["
    if row['arm'] in ('forced_prerequisites', 'forced_late', 'filler_forced'):
        ending += f"{row['target_value']}, {row['other_required']}, "
    if row['arm'] in ('filler', 'filler_forced'):
        full_count = len(tokenizer.encode(row['work'] + ending, add_special_tokens=False))
        return matched_padding(tokenizer, row['kept'], '\n' + ending, full_count)
    if row['arm'] == 'name_cue':
        text += 'Use v00_2 for the first requested value.\n'
    cue = 'Use v00_2 first and v01_2 second, in the requested order.\n'
    if row['arm'] == 'ordered_cue':
        text += cue
    if row['arm'] == 'neutral_cue':
        target = len(tokenizer.encode(text + cue + ending, add_special_tokens=False))
        return matched_padding(tokenizer, text, '\n' + ending, target)
    goal = 'The requested expression is ' + row['code'].splitlines()[-1] + '.\n'
    if row['arm'] == 'goal_cue':
        text += goal
    if row['arm'] == 'goal_neutral':
        target = len(tokenizer.encode(text + goal + ending, add_special_tokens=False))
        return matched_padding(tokenizer, text, '\n' + ending, target)
    return text + ending


def generations(root):
    return read_generation_rows(root) if (root / 'generation/generations.jsonl').exists() else []


def reduce_cell(root, rows):
    lookup = {r['id']: r for r in rows}
    groups = defaultdict(list)
    measurements = []
    for generation in generations(root):
        row = lookup[generation['sample_id']]
        prediction = parse_result(generation['produced_text'])
        valid = prediction is not None and len(prediction) == len(row['expected'])
        item = {k: row[k] for k in ('id', 'problem', 'horizon', 'consumer', 'arm', 'last_agrees', 'last_value', 'target_value')}
        item.update(prediction=prediction, parsed=valid, correct=prediction == row['expected'],
            output_arity=None if prediction is None else len(prediction),
            first_value_correct=prediction is not None and len(prediction) > 0 and prediction[0] == row['expected'][0],
            second_value_correct=prediction is not None and len(prediction) > 1 and prediction[1] == row['other_required'],
            second_is_first=valid and len(prediction) > 1 and prediction[1] == row['target_value'],
            forced_inputs=row['arm'] in ('forced_prerequisites', 'forced_late', 'filler_forced'),
            first_correct=valid and prediction[0] == row['expected'][0],
            first_is_last=valid and prediction[0] == row['last_value'],
            prerequisites_correct=valid and prediction[:2] == row['expected'][:2] if row['consumer'] in ('combine', 'copy') else None,
            internally_consistent=valid and prediction[2] == prediction[0] + prediction[1] if row['consumer'] == 'combine' else None,
            tokens=len(generation['generated_token_ids']))
        measurements.append(item)
        groups[(row['horizon'], row['consumer'], row['arm'], row['last_agrees'])].append(item)
    summary = {'expected': len(rows), 'observed': len(measurements), 'measurements': measurements,
        'groups': [dict(horizon=k[0], consumer=k[1], arm=k[2], last_agrees=k[3], n=len(v),
            correct=sum(r['correct'] for r in v), parsed=sum(r['parsed'] for r in v)) for k, v in groups.items()]}
    write_json(root / 'evaluation/summary.json', summary)
    return summary


def run(config, source, out, index, smoke):
    cell = prepare(config, out)['cells'][index]
    rows = cell['rows']
    if smoke:
        first = sorted({r['problem'] for r in rows})[:2]
        rows = [r for r in rows if r['problem'] in first and r['horizon'] == min(config['distractors'])]
    root = out / ('smoke' if smoke else 'cells') / cell['model'] / f"shard{cell['shard']}"
    model_config = yaml.safe_load((source / config['model_configs'][cell['model']]).read_text())['model']
    cfg = {'model': model_config, 'generation': {'max_new_tokens': 512, 'temperature': 0.,
        # The shared stopper waits for text after the match. Look ahead for
        # the closing bracket so it stops there, before another chat turn.
        'num_samples_per_item': 1, 'base_seed': config['seed'], 'stop_regex': r'Answer:\s*\[[\d,\s-]+(?=\])'},
        'capture': {'enabled': False}, 'prompt': {'mode': 'chat',
        'instruction': 'Use the supplied work if present. End with one line: Answer: [integers in result, in order]. Use integers, not expressions, in that final list.'}}
    write_json(root / 'config.json', cfg)
    model, tokenizer = load_hf_model_and_tokenizer(model_config)
    completed = {r['sample_id'] for r in generations(root)}
    contracts = []
    for ordinal, row in enumerate(rows):
        prefix = prefill(tokenizer, row)
        count = len(tokenizer.encode(prefix, add_special_tokens=False))
        contracts.append({'id': row['id'], 'prefix_tokens': count})
        if row['id'] in completed:
            continue
        run_config = dict(cfg, generation=dict(cfg['generation'], forced_prefix=prefix,
            max_new_tokens=count + (512 if row['arm'] == 'native' else 64)))
        generate_task(run_path=root, config=run_config, model=model, tokenizer=tokenizer,
            sample={'id': row['id'], 'question': row['question'], 'gold_answer': str(row['expected'])},
            sample_index=ordinal, sample_iter=0, progress=None, progress_label=row['id'])
    write_json(root / 'evaluation/token_contract.json', contracts)
    result = reduce_cell(root, rows)
    if result['observed'] != len(rows):
        raise ValueError('missing generations')
    # Check parsing and preservation of supplied inputs. A wrong third value
    # is an experimental outcome, including when the requested value is zero.
    control = ('copy', 'forced_prerequisites') if 'copy' in config.get('consumers', ()) else ('lookup', 'late')
    controls = [r for r in result['measurements'] if (r['consumer'], r['arm']) == control]
    ready = not smoke or (controls and all(r['parsed'] for r in controls) and
        (all(r['prerequisites_correct'] for r in controls) if control[0] == 'copy' else
         sum(r['correct'] for r in controls) >= len(controls) * .5))
    write_json(root / 'complete.json', {'complete': True, 'ready': bool(ready), 'smoke': smoke})
    if not ready:
        raise RuntimeError('easy program interface failed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--phase', choices=['prepare', 'smoke', 'run', 'reduce'], required=True)
    p.add_argument('--cell', type=int, default=0)
    a = p.parse_args(); config = yaml.safe_load(a.config.read_text())
    if a.phase == 'prepare':
        print(json.dumps({'cells': len(prepare(config, a.out)['cells'])}))
    elif a.phase in ('smoke', 'run'):
        run(config, a.source_root, a.out, a.cell, a.phase == 'smoke')
    else:
        for cell in json.loads((a.out / 'lock.json').read_text())['cells']:
            reduce_cell(a.out / 'cells' / cell['model'] / f"shard{cell['shard']}", cell['rows'])


if __name__ == '__main__':
    main()
