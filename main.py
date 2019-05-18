import os
import re
import sys
import time
from tkinter import Tk, filedialog
import populateDatabase as populateDB
from ModelUtils import sortDirFiles

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
                                                 ("CSV files", "*.csv"),
                                                 ("All files", "*.*")))
    sys.exit("No file uploaded, aborting script") if path == '' else print(path)
    if path[-2:] != 'gz':
        sys.exit("Wrong file extension, please upload .gz file")
    print(path)
    populateDB.populateDatabase(path)
    print("All operations complete")


elif ans == '2':
    print("Annual stores")
    batch = 'batch'
    app = Tk()
    app.withdraw()
    app.wm_attributes('-topmost', 1)
    dirPath = filedialog.askdirectory()
    sys.exit("No file uploaded, aborting script") if dirPath == '' else print(dirPath)
    dirPath = dirPath+'/'
    print(dirPath)

    file_list = sortDirFiles(dirPath)

    ans = input("Are you sure you want to upload files from this directory??\n"
                "Press Y or y to continue..\n"
                "Input: ")

    if ans == 'Y' or ans == 'y':
        ans = input("Select the number of files you wish to upload or for all files in directory type all: ")
        if ans == 'all' or ans == 'All':
            count = 1
            for file in file_list:
                path = dirPath + file
                populateDB.populateDatabase(path)
                print("File number {} complete".format(count))
                sys.exit("Operations complete exiting script")

        try:
            val = int(ans)
        except ValueError:
            sys.exit("That's not an integer! Exiting script")

        if int(ans) <= len(file_list):
            count = 1
            for file in file_list:
                path = dirPath + file
                populateDB.populateDatabase(path, batch)
                print("File number {} complete".format(count))
                if count == ans:
                    sys.exit("Operations complete exiting script")
    else:
        sys.exit("Operation aborted")

else:
    sys.exit("Invalid input")


