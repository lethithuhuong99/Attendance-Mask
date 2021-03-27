import os  # accessing the os functions
from Attendance_mask import Capture_Image
from Attendance_mask import Train_Image
from Attendance_mask import Recognize


# creating the title bar function

def title_bar():
    os.system('cls')  # for windows

    # title of the program

    print("\t**********************************************")
    print("\t***** Face Recognition Attendance System *****")
    print("\t**********************************************")


# creating the user main menu function

def mainMenu():
    title_bar()
    print()
    print(10 * "*", "WELCOME MENU", 10 * "*")
    print("[1] Capture Faces")
    print("[2] Train Images")
    print("[3] Recognize & Attendance")
    print("[4] Quit")

    while True:
        try:
            choice = int(input("Enter Choice: "))

            if choice == 1:
                CaptureFaces()
                break
            elif choice == 2:
                Trainimages()
                break
            elif choice == 3:
                RecognizeFaces()
                break
            elif choice == 4:
                print("Thank You")
                break
            else:
                print("Invalid Choice. Enter 1-4")
                mainMenu()
        except ValueError:
            print("Invalid Choice. Enter 1-4\n Try Again")
    exit


# ---------------------------------------------------------

# --------------------------------------------------------------
# calling the take image function form capture image.py file

def CaptureFaces():
    Capture_Image.takeImages()
    key = input("Enter any key to return main menu")
    mainMenu()


# -----------------------------------------------------------------
# calling the train images from train_images.py file

def Trainimages():
    Train_Image.TrainImages()
    key = input("Enter any key to return main menu")
    mainMenu()


# --------------------------------------------------------------------
# calling the recognize_attendance from recognize.py file

def RecognizeFaces():
    Recognize.recognize_attendence()
    key = input("Enter any key to return main menu")
    mainMenu()


# ---------------main driver ------------------
mainMenu()
