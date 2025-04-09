error = "Veillez entrer des nombres valides!"


def multiplication_of() -> int :
    number_1 = input('Entrez le premier nombre : ')
    number_2 = input('Entrez le deuxième nombre : ')

    try:
        resul = int(number_1) * int(number_2)
        print(resul)
    except ValueError:
        return error
    else:
        return resul
    
    
# def test_multiplication_of():
#     print("Use 3 and 4")
#     result = multiplication_of()
#     assert isinstance(result, int)
#     assert result == 12

#     print()

#     print("Use invalid value")
#     result = multiplication_of()
#     assert isinstance(result, str)
#     assert result == error


if __name__ == "__main__":
    multiplication_of()