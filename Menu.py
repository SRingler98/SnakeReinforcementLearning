# Start menu to make the program easier to work with

def top_menu():
    print("[1] Play Snake")
    print("[2] RL Agent")
    print("[Q] Quit")


def menu():
    """
    This is a text-based menu that helps organize functionality.
    """
    do: top_menu()
    option = int(input("Enter your selection: "))

    while option != 0:
        if option == 1:
            #play snake
            print("Snake")
            
            pass
        elif option == 2:
            #display RL agent menu
            print("RL")
            pass
        elif option == 0:
            #Quit condition
            print("Quitting...")
            break
        else:
            print("Invalid input.")
        
        top_menu()
        option = input("Enter your selection: ")

menu()