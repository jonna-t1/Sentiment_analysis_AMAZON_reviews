import os
import re
import sys


def sortDirFiles(dirPath):
    dirFiles = sorted(os.listdir(dirPath))  # list of directory files

    # re.match(pattern, string, flags=0)
    numberOfFiles = len(dirFiles)
    numberOfDigits = len(str(numberOfFiles))
    if numberOfDigits > 2:
        sys.exit("Please enter less than 100 files")

    fileArray = [[], []]

    for i in dirFiles:
        # print(i)
        m = re.search("^.*?\([^\d]*(\d+)[^\d]*\).*$", i)
        if m:
            # sortedArray.append(i)
            strDig = m.groups()[0]

            if len(strDig) == 1:
                fileArray[0].append(i)

            if len(strDig) == 2:
                fileArray[1].append(i)
        else:
            sys.exit("Incorrect file naming convention")

    file_list = [item for sublist in fileArray for item in sublist]
    print(file_list)
    return file_list

def getModelFileName(path):
    dirFiles = os.listdir(path)  # list of directory files
    lastFilename = dirFiles[-1]

    numbersInFile = [int(x) for x in lastFilename if x.isdigit()]

    # guards against an empty directory
    if numbersInFile == [] or (lastFilename == 'finalised_model.sav' or lastFilename == 'finalised_tfidftransformer.sav'):
        if lastFilename == 'finalised_model.sav':
            return path+'retrained_model.sav1'
        if lastFilename == 'finalised_tfidftransformer.sav':
            return path+'retrained_tfidftransformer.sav1'
    elif len(numbersInFile) > 1:
        print("An error has occured writing to model to file")
        return "Error"

    newFileNo = numbersInFile[0] + 1
    file = path + lastFilename[:-1] + str(newFileNo)

    return file


def getLastModel(path):
    dirFiles = os.listdir(path)  # list of directory files

    if dirFiles == []:
        sys.exit("Empty Directory, please include the inital model")

    lastFilename = dirFiles[-1]
    lastFilename = path + lastFilename
    return lastFilename