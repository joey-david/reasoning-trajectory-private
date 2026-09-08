from src.experiments.completed_work import variants, verified_prefix, prefill
from src.experiments.state_routing_heads import build_head_cases


def test_prefix_gate_does_not_read_final_answer():
    case = build_head_cases(screen_count=0, development_count=0, test_count=1, seed=4)[0]
    working = case['clean']['text'].split('Solution:\n')[1]
    prefix = '\n'.join(working.splitlines()[:3])
    a = {'generation': {'text': prefix + '\nAnswer=0'}}
    b = {'generation': {'text': prefix + '\nAnswer=9'}}
    assert verified_prefix(case, a) == verified_prefix(case, b)
    assert verified_prefix(case, a) is not None


def test_distractor_horizon_never_updates_target():
    case = build_head_cases(screen_count=0, development_count=0, test_count=1, seed=4)[0]
    prefix = '\n'.join(case['clean']['text'].split('Solution:\n')[1].splitlines()[:3]) + '\n'
    rows = variants(case, prefix, 123)
    assert {r['expected'][0] for r in rows} == {case['clean_answer']}
    late = next(r for r in rows if r['arm'] == 'target_late')
    text = prefill(None, late)
    # Event 2 can feed distractor descendants and must remain before them.
    lines = text.splitlines()
    original = prefix.splitlines()
    assert lines[0].split('. ', 1)[1] == original[1].split('. ', 1)[1]
    assert lines[-3].split('. ', 1)[1] == original[0].split('. ', 1)[1]
    assert lines[-2].split('. ', 1)[1] == original[2].split('. ', 1)[1]
