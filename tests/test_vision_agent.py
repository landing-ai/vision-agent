from vision_agent.agent.vision_agent import sample_n_evenly_spaced


def test_sample_n_evenly_spaced_side_cases():
    # Test for empty input
    assert sample_n_evenly_spaced([], 0) == []
    assert sample_n_evenly_spaced([], 1) == []

    # Test for n = 0
    assert sample_n_evenly_spaced([1, 2, 3, 4], 0) == []

    # Test for n = 1
    assert sample_n_evenly_spaced([1, 2, 3, 4], -1) == []
    assert sample_n_evenly_spaced([1, 2, 3, 4], 5) == [1, 2, 3, 4]


def test_sample_n_evenly_spaced_even_cases():
    assert sample_n_evenly_spaced([1, 2, 3, 4, 5, 6], 2) == [1, 6]
    assert sample_n_evenly_spaced([1, 2, 3, 4, 5, 6], 3) == [1, 3, 6]
    assert sample_n_evenly_spaced([1, 2, 3, 4, 5, 6], 4) == [1, 3, 4, 6]
    assert sample_n_evenly_spaced([1, 2, 3, 4, 5, 6], 5) == [1, 2, 3, 5, 6]
    assert sample_n_evenly_spaced([1, 2, 3, 4, 5, 6], 6) == [1, 2, 3, 4, 5, 6]


def test_sample_n_evenly_spaced_odd_cases():
    assert sample_n_evenly_spaced([1, 2, 3, 4, 5, 6, 7], 2) == [1, 7]
    assert sample_n_evenly_spaced([1, 2, 3, 4, 5, 6, 7], 3) == [1, 4, 7]
    assert sample_n_evenly_spaced([1, 2, 3, 4, 5, 6, 7], 4) == [1, 3, 5, 7]
    assert sample_n_evenly_spaced([1, 2, 3, 4, 5, 6, 7], 5) == [1, 3, 4, 5, 7]
    assert sample_n_evenly_spaced([1, 2, 3, 4, 5, 6, 7], 6) == [1, 2, 3, 5, 6, 7]
    assert sample_n_evenly_spaced([1, 2, 3, 4, 5, 6, 7], 7) == [1, 2, 3, 4, 5, 6, 7]
