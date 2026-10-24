from repository.repository import rrf


def test_rrf_both_lists():
    results = rrf([['a', 'b', 'c', 'd'], ['b', 'd']], weights=[0.5, 0.5])
    assert results == ['b', 'd', 'a', 'c']
    

def test_rrf_ignore_second_list():
    result = rrf([['a', 'b', 'c', 'd'],['d', 'e', 'f']], weights=[1.0, 0])
    assert result[:4] == ['a', 'b', 'c', 'd']
