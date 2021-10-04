
from DisplayGrid import DisplayGrid


DG = DisplayGrid(10, 50)

state_tuple = (4, 4)

while 1:
    DG.draw_grid()
    DG.event_handler(state_tuple)
