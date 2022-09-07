def return_ore_secondi(n):
    ora = n // 3600
    min = int((n % 3600)/60)
    return print(f'{ora}h {min}s')


