import glob
import cv2

from imgrecog import recognizer


def main():
    number = 0
    for file in sorted(glob.iglob('resources/originals/*')):
        img = cv2.imread(file)
        mask = cv2.imread('resources/mask.jpg')
        recognizer.recognize(mask, img, number)
        number = number + 1


if __name__ == '__main__':
    main()
