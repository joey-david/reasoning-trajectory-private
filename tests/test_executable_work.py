from src.experiments.executable_work import cases
from src.experiments.dependency_state import program_facts


def test_every_arm_has_the_same_executable_oracle_and_keeps_prerequisites():
    rows = cases({'seed': 8, 'problems': 3, 'distractors': [2, 8, 32]})
    for row in rows:
        values = program_facts(row['code'])['values']
        assert list(values['result']) == row['expected']
        assert f"v00_2 = {values['v00_2']}" in row['kept']
        assert f"v01_2 = {values['v01_2']}" in row['kept']
        if row['arm'] == 'late':
            assert row['work'].splitlines()[-1] == f"v00_2 = {values['v00_2']}"
        if row['arm'] not in ('last_name_hidden', 'last_value_hidden'):
            for line in row['work'].splitlines():
                name, value = line.split(' = ')
                assert int(value) == values[name]


def test_longer_histories_leave_the_requested_values_unchanged():
    rows = cases({'seed': 2, 'problems': 4, 'distractors': [2, 8, 32]})
    for index in range(4):
        for consumer in ('lookup', 'combine'):
            assert len({tuple(r['expected']) for r in rows if r['problem'] == index and r['consumer'] == consumer}) == 1


def test_copy_and_combine_share_inputs_and_three_value_answers():
    config = dict(seed=8, problems=4, distractors=[2, 8], consumers=['copy','combine'],
                  arms=['early','late','both_late','forced_prerequisites','forced_late','arithmetic_only'])
    rows = cases(config)
    for row in rows:
        assert len(row['expected']) == 3
        assert row['expected'][:2] == [row['target_value'], row['other_required']]
        assert list(program_facts(row['code'])['values']['result']) == row['expected']
        if row['arm'] == 'both_late':
            assert row['work'].splitlines()[-4].startswith('v00_2 = ')
            assert row['work'].splitlines()[-1].startswith('v01_2 = ')
        if row['arm'] == 'arithmetic_only':
            assert len(row['work'].splitlines()) == 2


def test_forced_inputs_and_padding_do_not_change_the_requested_result():
    from src.experiments.executable_work import prefill
    class CharTokenizer:
        def encode(self, text, add_special_tokens=False): return list(map(ord, text))
        def decode(self, tokens): return ''.join(map(chr, tokens))
    tokenizer = CharTokenizer()
    rows = cases(dict(seed=8, problems=1, distractors=[8], consumers=['combine'],
                     arms=['early','filler','ordered_cue','neutral_cue','forced_prerequisites','filler_forced']))
    prefixes = {r['arm']: prefill(tokenizer, r) for r in rows}
    assert len(prefixes['early']) == len(prefixes['filler'])
    assert len(prefixes['ordered_cue']) == len(prefixes['neutral_cue'])
    assert len(prefixes['forced_prerequisites']) == len(prefixes['filler_forced'])
    for arm in ('forced_prerequisites', 'filler_forced'):
        assert prefixes[arm].endswith(f"Answer: [{rows[0]['target_value']}, {rows[0]['other_required']}, ")
