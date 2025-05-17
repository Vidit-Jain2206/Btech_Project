print('babli')
import math
import random

from scipy import density_vehicle

class Lane:
    def __init__(self,cars : int , signal_state : chr , max_wait : int , waiting :int, name : str, emergency = False) -> None:
        self.cars = cars
        self.signal_state =signal_state
        self.max_wait = max_wait
        self.waiting = waiting
        self.name = name
        self.emergency = emergency

    def increase_car(self) :
        self.cars += random.randint(0,3)
        if self.signal_state == 'R' :
            self.waiting += 1

    def green(self):
        self.signal_state ='G'
        self.waiting = 0
        self.cars -= 2  #depends on speed of car passing the jt 
    
    def yellow(self):
        print("yellow for : ",self.name)
        pass

    def emergency_vehicle(self):
        print("jaane nhi denge tujhe")
        pass

def myfunction(e) :
    return e.waiting

l1 = Lane(20,'G',4,0,'DEL')
l2 = Lane(15,'R',5,5,'BOM')
l3 = Lane(10,'R',6,6,'PAK')
l4 = Lane(18,'R',8,8,'GUJ')
junction = (l1,l2,l3,l4)        #priority queue
lane_waiting_reached = []
global temp
temp = l1
current =l1
for i in range(4) :
    wtime = 0   
    for lane in junction:
        lane.increase_car()
        print(lane.name,lane.cars,lane.signal_state,lane.waiting)
        if lane.emergency == True :
            temp = lane
            lane.emergency_vehicle()
            break
        
        elif lane.cars > temp.cars:
            temp = lane
        elif lane.waiting >= lane.max_wait:
            # temp = lane 
            lane_waiting_reached.append(lane)
            # break

    lane_waiting_reached.sort(reverse = True, key = myfunction)
    print(lane_waiting_reached[0].name)
    temp = lane_waiting_reached[0]
    if temp !=current :
        current.yellow()  #transitioning from RED TO YELLOW
        temp.yellow()
        current.signal_state ='R'
    else:
        pass

    
    current= temp
    print("signal changes to",current.name)
    current.green()
    print(current.name,current.cars)
    
