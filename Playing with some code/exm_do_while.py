def main():
    print("Digite um numero:")
    a = int(input())
    c = 1

    while(c == 1):
        print("Número digitado: ", a)

        print("Continuar?(1 - Sim/ 0 - Não):")
        c = int(input())
        if(c == 0):
            break
        print("Digite outro numero:")
        a = int(input())


main()