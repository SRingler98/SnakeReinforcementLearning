# Start menu to make the program easier to work with
from SnakeEngine import *
from QLearning import *
from Graphing import *

def welcome_message():
    """
    Prints the program start message.
    """
    menu_line = "==================================================="
    print(menu_line)
    print("\t\tWelcome to SnakeRL!")
    print(menu_line)
    print("Authors: Shawn Ringler and Frankie Torres")
    print()

def top_menu():
    """
    Prints the top level selection menu.
    """
    menu_line = "-----------------------------"
    print(menu_line)
    print("\t[1] Play Snake")
    print("\t[2] RL Agent")
    print()
    print("\t[0] Quit")
    print(menu_line)
    print()


def menu():
    """
    This is a text-based menu that contains and organizes the entire project's functionality.
    """
    
    welcome_message()
    top_menu()

    #Get user input
    try:
        option = input("Enter your selection: ")
        print() #newline
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt. Quitting program...\n")
        quit()
    except:
        print("Error!\n")

    #Try to convert to int
    try:
        option = int(option)
    except ValueError:
        option == -1  #setting to unused value in order to pass below condition checks

    #While the menu is still running (user hasn't entered '0')
    while option != 0:

        #Try to convert to int
        try:
            option = int(option)
        except ValueError:
            option == -1  #setting to unused value in order to pass below condition checks

        #Make user selection
        if option == 1:
            #play snake
            print("Snake")
        elif option == 2:
            #display RL agent menu
            print("RL")
        else:
            print("Invalid input.\n")

        print() #newline
        top_menu()

        #Get user input
        try:
            option = input("Enter your selection: ")
            print() #newline
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt. Quitting program...\n")
            quit()
        except:
            print("Error!\n")

        #try to convert to integer
        try:
            option = int(option)
        except ValueError:
            option == -1  #setting to unused value in order to pass below condition checks

menu()