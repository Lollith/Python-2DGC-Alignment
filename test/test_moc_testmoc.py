from testmoc import multiplication_of


def test_multiplication_of():
    print("Use 3 and 4")
    result = multiplication_of()
    assert isinstance(result, int)
    assert result == 12

    print()
    
    print("Use invalid value")
    result = multiplication_of()
    assert isinstance(result, str)
    assert result == "error"
# https://medium.com/@romualdoluwatobi/unit-testing-python-mock-object-comprendre-avec-deux-exemples-simples-5f8b21cb0816