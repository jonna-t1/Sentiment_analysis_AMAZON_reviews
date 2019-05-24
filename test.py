from pathlib import Path
import os
import re

def sortDirFiles():
    p = Path(os.getcwd())
    dirPath = p / ('savedModels/model/')
    dirFiles = os.listdir(dirPath)  # list of directory files
    # print(dirFiles)

    numberOfFiles = len(dirFiles)
    numberOfDigits = len(str(numberOfFiles))

    if numberOfFiles > 9:

        fileArray = [[], []]

        for i in dirFiles:
            # print(type(i))

            num = [int(char) for char in i if char.isdigit()]

            # num = str(num)
            print(num)
            if len(num) == 1:
                fileArray[0].append(i)

            if len(num) == 2:
                fileArray[1].append(i)

        file_list = [item for sublist in fileArray for item in sublist]
        return file_list

    if numberOfFiles < 9:
        return dirFiles.sort()

print(sortDirFiles())