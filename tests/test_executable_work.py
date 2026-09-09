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
