# Start menu to make the program easier to work with
from matplotlib.pyplot import get
from SnakeEngine import *
from QLearning import *
from Graphing import *

def print_welcome_message():
    """
    Prints the program start message.
    """
    menu_line = "==================================================="
    print(menu_line)
    print("\t\tWelcome to SnakeRL!")
    print(menu_line)
    print("     Authors: Shawn Ringler and Frankie Torres")


def print_menu():
    """
    Prints the top level selection menu.
    """
    title_line = "-----------------Main Menu-----------------"
    menu_line  = "-------------------------------------------"
    print()
    print(title_line)
    print("\t[1] Play Snake")
    print("\t[2] Reinforcement Learning")
    print()
    print("\t[0] Quit")
    print(menu_line)


def print_rl_menu():
    """
    Prints the Reinforcement Learning menu
    """
    print()
    title_line = "---------------RL Menu---------------"
    menu_line  = "-------------------------------------"
    print(title_line)
    print("\t[1] Demo")
    print("\t[2] Train Agent")
    print("\t[3] Graph Performance")
    print()
    print("\t[0] Back to Main Menu")
    print(menu_line)


def demo():

    #Show agent training
    agent = QLearning(100,
                debug_on=False,
                visuals_on_while_training=True,
                load_on=False,
                save_on=False,
                file_name="",
                show_score_plot=False,
                training_on=True)

    agent.run_optimal_game()

    agent = None
    

def rl_menu():
    """
    This is the Reinforcement Learning sub-menu running loop. This lets the user
    select options related to Reinforcement Learning.
    """
    print_rl_menu()
    option = get_user_input()
    option = handle_conversion_to_int(option)

    while option != 0:
        option = handle_conversion_to_int(option)

        #Make user selection
        if option == 1:
            #[1] Demo 
            print("'[1] Demo' selected. Starting Demo...")
            demo()
        elif option == 2:
            #[2] Train Agent
            print("'[2] Train Agent' selected. Loading training menu...")
        else:
            print("Invalid input.")

        print_rl_menu()
        option = get_user_input()
        option = handle_conversion_to_int(option)   #needed for while condition evaluation
    
    print("Going back to main menu...")


def get_user_input():
    #Gets the user input while avoiding
    try:
        user_input = input("Enter your selection: ")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt. Quitting program...\n")
        quit()
    except:
        print("Error!\n")
    return user_input


def handle_conversion_to_int(user_input):
    #Try to convert to int
    try:
        converted_user_input = int(user_input)
        return converted_user_input
    except ValueError:
        return user_input


def menu():
    """
    This is a text-based menu that contains and organizes the entire project's functionality.
    """
    print_welcome_message()
    print_menu()
    option = get_user_input()
    option = handle_conversion_to_int(option)

    #While the menu is still running (user hasn't entered '0')
    while option != 0:
        option = handle_conversion_to_int(option)

        #Make user selection
        if option == 1:
            #Play snake
            print("'[1] Play Snake' selected. Loading Snake game...")
            SnakeEngine(10).run_game_in_real_time()
        elif option == 2:
            #Run the Reinforcement Learning sub-menu
            print("'[2] Reinforcement Learning' selected. Loading RL menu...")
            rl_menu()
        else:
            print("Invalid input.")

        print_menu()
        option = get_user_input()
        option = handle_conversion_to_int(option)   #needed for while condition evaluation


menu()  #testing the menu. DELETE LATER TO CALL IN MAIN INSTEAD OF HERE ONCE FINISHED