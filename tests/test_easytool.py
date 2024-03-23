from vision_agent.agent.easytool import topological_sort


def test_basic():
    tasks = [
        {"id": 1, "dep": [-1]},
        {"id": 3, "dep": [2]},
        {"id": 2, "dep": [1]},
    ]
    assert topological_sort(tasks) == [tasks[0], tasks[2], tasks[1]]


def test_two_start():
    tasks = [
        {"id": 1, "dep": [-1]},
        {"id": 2, "dep": [1]},
        {"id": 3, "dep": [-1]},
        {"id": 4, "dep": [3]},
        {"id": 5, "dep": [2, 4]},
    ]
    assert topological_sort(tasks) == [tasks[0], tasks[2], tasks[1], tasks[3], tasks[4]]


def test_broken():
    tasks = [
        {"id": 1, "dep": [-1]},
        {"id": 2, "dep": [3]},
        {"id": 3, "dep": [2]},
        {"id": 4, "dep": [3]},
    ]

    assert topological_sort(tasks) == [tasks[0], tasks[1], tasks[2], tasks[3]]
