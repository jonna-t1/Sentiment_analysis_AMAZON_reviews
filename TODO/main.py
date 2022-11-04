import os
import re
import sys
import time
from tkinter import Tk, filedialog
import populateDatabase as populateDB
from ModelUtils import sortDirFiles


def main():
    batch = ''

    ans = input("What data would you like to upload??:\n"
                   "Individual file use:    1\n"
                   "Annual datastore use:   2\n"
                   "Input:  ")

    if ans == '1':
        print("Individual file")
        app = Tk()
        app.withdraw()
        app.wm_attributes('-topmost', 1)
        path = filedialog.askopenfilename(filetypes=(("Zipped files", "*.gz"),
                                                     ("All files", "*.*")))

        app.destroy()
        sys.exit("No file uploaded, aborting script") if path == '' else print(path)
        if path[-2:] != 'gz':
            sys.exit("Wrong file extension, please upload .gz file")
        print(path)
        modelFiles = []
        modelFiles = populateDB.populateDatabase(path, modelFiles)
        print("All operations complete")

        print('Files added...')
        for i in modelFiles:
            print(i)


    elif ans == '2':
        print("Annual stores")
        batch = 'batch'
        app = Tk()
        app.withdraw()
        app.wm_attributes('-topmost', 1)
        dirPath = filedialog.askdirectory()
        app.destroy()
        sys.exit("No file uploaded, aborting script") if dirPath == '' else print(dirPath)
        dirPath = dirPath+'/'
        print(dirPath)

        file_list = sortDirFiles(dirPath)

        ans = input("Are you sure you want to upload files from this directory??\n"
                    "Press Y or y to continue..\n"
                    "Input: ")

        if ans == 'Y' or ans == 'y':
            batch = 'batch'
            ans = input("How many files do you wish to upload - there are {} files: ".format(len(file_list)))
            try:
                val = int(ans)
            except ValueError:
                sys.exit("That's not an integer! Exiting script")

            modelFiles = []

            if int(ans) <= len(file_list):
                count = 1
                for file in file_list:
                    print("File upload {} started".format(count))
                    path = dirPath + file
                    filePath = populateDB.populateDatabase(path, modelFiles)
                    modelFiles.append(filePath)
                    print("File number {} complete".format(count))
                    print("File completed: "+ file)
                    if int(ans) == count:
                        break
                    count += 1
                print('Files added...')
                for i in modelFiles:
                    print(i)
            else:

                sys.exit("Number exceeds the number of files in directory. Exiting script...")

        else:
            sys.exit("Operation aborted")

    else:
        sys.exit("Invalid input")

main()
# if __name__ == '__main__':
#     main()

