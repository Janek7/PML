import os

def main():
    while True:
        input_text = input('cmd?')

        if input_text == 'Hello':
            print('Antwort')
        elif input_text == 'Exit':
            exit()
        else:
            try:
                os.system(input_text)
            except Exception as e:
                print(e)
        print()

if __name__ == '__main__':
    main()