from src.experiments.completed_work import variants, verified_prefix, prefill, answer_digit
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


def test_answer_annotations_are_distinct_from_intermediate_values():
    assert answer_digit("Answer: 6 (Cora's final score)") == [6]
    assert answer_digit('Answer: 6.') == [6]
    assert answer_digit('Answer: [6]') == [6]
    assert answer_digit('Cora has 6.') is None
    assert answer_digit('Answer: 6 + 1 = 7') is None


def test_last_distractor_agreement_is_controlled_before_generation():
    case = build_head_cases(screen_count=0, development_count=0, test_count=1, seed=4)[0]
    prefix = '\n'.join(case['clean']['text'].split('Solution:\n')[1].splitlines()[:3]) + '\n'
    for agrees in (False, True):
        for row in variants(case, prefix, 123, agrees):
            if row['horizon']:
                value = int(row['oracle_history'].splitlines()[-1].rsplit(' = ', 1)[1].strip('.'))
                assert (value == row['expected'][0]) == agrees
