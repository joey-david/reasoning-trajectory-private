from reasoning_trajectory.extract.answers import answer_correct, extract_final_answer, normalize_number


def test_numeric_answer_extraction() -> None:
    assert normalize_number("$1,200.00") == "1.2E+3"
    assert extract_final_answer("Step 1: add.\nFinal answer: 17 dollars") == "17"
    assert answer_correct("Final answer: 42.", "42") == ("42", True)
    assert answer_correct("Final answer: 41", "42") == ("41", False)
