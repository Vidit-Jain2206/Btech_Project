
import sys
import os
import random
import math

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from VehicleDetection.script import detect_vehicles
import threading

path = "/Users/vidit2003/Desktop/BtechProject"
pictures = [f'{path}/TrafficLightsPhotos/pic1.jpg', f'{path}/TrafficLightsPhotos/pic2.jpg', f'{path}/TrafficLightsPhotos/pic3.jpg', f'{path}/TrafficLightsPhotos/pic5.jpg', f'{path}/TrafficLightsPhotos/pic1.jpg', f'{path}/TrafficLightsPhotos/pic2.jpg', f'{path}/TrafficLightsPhotos/pic3.jpg', f'{path}/TrafficLightsPhotos/pic5.jpg']

class Lane:
    def __init__(self,cars : int , signal_state : chr , max_wait : int , waiting :int, name : str, emergency = False) -> None:
        self.cars = cars
        self.signal_state = signal_state
        self.max_wait = max_wait
        self.waiting = waiting
        self.name = name
        self.emergency = emergency


    def green(self):
        self.signal_state ='G'
        self.waiting = 0
        print("green for : ",self.name)
    
    def yellow(self):
        self.signal_state ='Y'
        print("yellow for : ",self.name)
        pass

    def emergency_vehicle(self):
        print("jaane nhi denge tujhe")
        pass
    def __eq__(self, other):
        if isinstance(other, Lane):
            return self.name == other.name
        return False

def myfunction(e) :
    return e.waiting

# index variables will be used to access the pictures as per the lane after every green signal, abcd stores the current picture of lanes, so here I stimulate only 2 green signals
# 0 -> a,b,c,d
# 1 -> e,f,g,h  
# 2 -> a,b,c,d
# 3 -> e,f,g,h

def calculate_vehicle_count(junction,index):
    for i in range(len(junction)) :
        image_index = random.randint(0,7)
        print(pictures[image_index])
        print("detecting vehicles")
        total_vehicles, category,image = detect_vehicles(pictures[image_index])
        # print("detected vehicles",count)
        lane = junction[i]
        lane.cars = total_vehicles
        print("no of vehciles in ",i+1,"lane",total_vehicles,category)

def increase_waiting_time(junction):
    for lane in junction:
        if lane.signal_state == 'G':
            lane.waiting = 0
        else:
            lane.waiting += 1
    
def stimulate_junction(junction,i,current):
     # calculate vehicle count for each lane
    lane_waiting_reached = []
    calculate_vehicle_count(junction,i)
    temp = junction[0]
    for lane in junction:
        if lane.emergency == True :
            temp = lane
            lane.emergency_vehicle()
            break
        
        if lane.cars > temp.cars:
            temp = lane
        if lane.waiting >= lane.max_wait:
            # temp = lane 
            lane_waiting_reached.append(lane)
            # break

    if(len(lane_waiting_reached) > 0):
        lane_waiting_reached.sort(reverse = True, key = myfunction)
        temp = lane_waiting_reached[0]
        lane_waiting_reached.clear()
   
    
    if current == None:
        current = temp
        current.signal_state = 'G'
        current.green()

    elif temp == current:
        # signal does not change, current signal still have higher number of vehicles
        print("no change in signal")
    else:
        # transition to yellow from green
        if(current != None): current.yellow()
        # temp.yellow()
        current.signal_state ='R'
        current = temp
        current.green()

    # increase the waiting time of all lanes
    increase_waiting_time(junction)
    green_time = math.floor(current.cars / 2)

    return green_time,current



# Assume these functions and classes are already defined
# stimulate_junction(junction, i, current_green_signal) -> (green_time, current)
# Lane class is defined somewhere



def runs(junction, i=0, current_green_signal=None):
    print(f"\n▶️ Iteration {i}")
    green_time, current = stimulate_junction(junction, i, current_green_signal)
    print(f"⏱️ Next run in {green_time} seconds")

    # Schedule the next execution dynamically
    timer = threading.Timer(green_time, runs, args=(junction, i + 1, current))
    timer.start()


# Create lanes and junction
l1 = Lane(0, 'R', 4, 0, 'DEL')
l2 = Lane(0, 'R', 5, 0, 'BOM')
l3 = Lane(0, 'R', 6, 0, 'PAK')
l4 = Lane(0, 'R', 8, 0, 'GUJ')
junction = (l1, l2, l3, l4)

# Start execution
runs(junction)




   
    
    
