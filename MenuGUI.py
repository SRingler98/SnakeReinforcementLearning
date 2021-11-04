# Start menu to make the program easier to work with
from tkinter import font
from tkinter.constants import N
import PySimpleGUI as sg
from SnakeEngine import *
from QLearning import *
from Graphing import *
import time
import os
import webbrowser

#Demos
def midterm_demo():
    #---Train an agent on 100 episodes---#
    learning_demo = QLearning(100,
               debug_on=False,
               visuals_on_while_training=True,
               load_on=False,
               save_on=False,
               file_name="",
               show_score_plot=False,
               training_on=True)

    learning_demo.run_optimal_game(n_times=1)

    #---Demo with 10mil table---#
    tenMilDemo = QLearning(50,
               debug_on=True,
               visuals_on_while_training=True,
               load_on=True,
               save_on=False,
               file_name="saved_tables/10mil.json",
               show_score_plot=False,
               training_on=True)

    tenMilDemo.run_optimal_game(n_times=1)

    #--Graph one agent optimals--#
    graphing_agent = QLearning(100,
               debug_on=False,
               visuals_on_while_training=True,
               load_on=False,
               save_on=False,
               file_name="saved_tables/10mil.json",
               show_score_plot=False,
               training_on=True)

    graphing_agent.learning_loop_create_graphs(5) #This graphs the agents performance over n optimal runs per ep

    #---Multiple Agent Graph Demo---#
    learning_loop_multiple_agents_create_graph(100, 10, 10)


saved_tables = os.listdir("saved_tables")
               
#sg.theme('TealMono')  #Window colors
sg.theme('SandyBeach')  #Window colors
sg.theme('LightBrown13')  #Window colors

#------Menu----------#
menu_def = [ ['&menuButton', ['&opt1', '&opt2']] ]

#------Columns-------#

layout = [  

    #Title frame
    [sg.Frame(layout=[
        [sg.Text("SnakeRL", size=(20, 1), justification="center", font=("Helvetica", 38, 'bold'), text_color="Green")]], 
            title='')],

    #Header buttons frame
    [sg.Frame(layout=[
        [sg.Button('Play\nSnake', size=(10, 1), font=("Helvetica", 22)), 
            sg.Button('Midterm\nDemo', size=(10, 1), font=("helvetica", 22)),
            sg.Button('Visit\nGithub', size=(10, 1), font=("Helvetica", 22))]], 
            title='')],  

    #Q-Learning frame
    [sg.Frame(layout=[
        #title
        [sg.Text('Q-Learning:', size=(28, 1), font=("Helvetica", 26, 'bold'), text_color="Green")],
        #Training frame
        [sg.Frame(layout=[
            [sg.Text('Episodes:', size=(8, 1), font=("Helvetica", 14)),
                sg.InputText('1', size=(8, 1), font=("Helvetica", 14))],
            [sg.CBox('Visualize Training', size=(16, 1), font=("Helvetica", 14))],
            [sg.CBox('Debug Mode', size=(16, 1), font=("Helvetica", 14))],
            [sg.Radio('Run without loading or saving', "RADIO1", size=(30, 1), font=("Helvetica", 14))],
            [sg.Radio('Load Q-table', "RADIO1", size=(13, 1), font=("Helvetica", 14)),
                sg.Combo(saved_tables, size=(31, 1), font=("Helvetica", 15))
                ],
            [sg.Radio('Save Q-table', "RADIO1", size=(13, 1), font=("Helvetica", 14)),
                sg.InputText('', size=(32, 1), font=("Helvetica", 15))
                ],
            [sg.CBox('Run optimally after training', size=(30, 1), font=("Helvetica", 14))],
            [sg.CBox('Create score graph', size=(30, 1), font=("Helvetica", 14))],
            [sg.Button('Train', size=(46, 1), font=("Helvetica", 16))],
            ],
            title='Training an Agent',
            title_location="n",
            border_width=3,
            font=("Helvetica", 20))
            ], 

        #Graphing frame

        [sg.Frame(layout=[
            [sg.Text('Episodes:', size=(16, 1), font=("Helvetica", 14)),
                sg.InputText('1', size=(8, 1), font=("Helvetica", 14))],
            [sg.Text('Agent to Train:', size=(16, 1), font=("Helvetica", 14)),
                sg.InputText('1', size=(8, 1), font=("Helvetica", 14))],
            [sg.Text('Optimal Runs per episode:', size=(16, 1), font=("Helvetica", 14)),
                sg.InputText('1', size=(8, 1), font=("Helvetica", 14))],
            [sg.Button('Graph', size=(46, 1), font=("Helvetica", 16))]
            ],
        title='Graphing Performance',
        title_location="n",
        border_width=3,
        font=("Helvetica", 20))]
        ], 

        title='')], 
    
]   #end of layout

window = sg.Window("SnakeRL", layout)

#------Pygame+PySimpleGUI integration-----#
#graph = window['-GRAPH-']
#embed = graph.TKCanvas
#print(embed)
#os.environ['SDL_WINDOWID'] = str(embed.winfo_id())
#os.environ['SDL_VIDEODRIVER'] = 'windib'

while True:
    event, vals = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': #window is being closed by user
        break
    if event == 'Play\nSnake': #start Snake game for human player
        SnakeEngine(10).run_game_in_real_time()
    elif event == 'Visit\nGithub':   #opens the github repo
        webbrowser.open('https://github.com/frankietorres/SnakeRL')
    elif event == 'Midterm\nDemo': #Demo our midterm progress
        midterm_demo()
    elif event == 'Train':
        eps = int(vals[0])
        visuals = vals[1]
        debug = vals[2]
        loading = vals[4]
        loadfile = vals[5]
        saving = vals[6]
        savefile = vals[7]
        optimalrun = vals[8]
        graph_scores = vals[9]

        file = ""
        if loading == True:
            file = "saved_tables/" + loadfile
        elif saving == True:
            file = "saved_tables/" + savefile + ".json"

        agent = QLearning(eps,
            debug_on=debug,
            visuals_on_while_training=visuals,
            load_on=loading,
            save_on=saving,
            file_name=file,
            show_score_plot=False,
            training_on=True)

        if optimalrun == True:
            agent.run_optimal_game(n_times=1)

        if graph_scores == True:
            agent.learning_loop_create_graphs(1)

        print(event, vals)

    elif event == 'Graph':
        eps = int(vals[10])
        agents = int(vals[11])
        optimals = int(vals[12])
        learning_loop_multiple_agents_create_graph(eps, agents, optimals)
        
    else:
        pass
    


window.close()
