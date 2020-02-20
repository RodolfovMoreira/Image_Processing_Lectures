
def main():
    lista = input().split(' ')
    a = int(lista[0])
    b = int(lista[1])

    if(a >= b):
        print(b,a)
    else:
        print(a,b)
main()