from signLanguageRecognizer import signLanguageRecognizer
from createDataset import createDataset


def handleActionInMain():
    while True:
        _input = input('1 - Create new class \n2 - Sign Language Recognizer\nq - Quit\n')
        if _input == '1':
            createDataset()
        if _input == '2':
            signLanguageRecognizer()
            pass
        elif _input == 'q':
            print('\nLeaving a program...')
            break
        else:
            pass


handleActionInMain()
