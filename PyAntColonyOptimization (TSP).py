#ANT COLONY OPTIMIZATION FOR THE TRAVELING SALESMAN PROBLEM - ANG WEIHENG 10C
import turtle as t #visualization
import math as m
import random as r
import time as tm
import copy as c
import matplotlib.pyplot as mp
tscreen = t.Screen()
tscreen.bgcolor("black")
tscreen.tracer(0)
t.hideturtle()
t.color("white")
t.up()
animate_best_path = t.Turtle()
animate_best_path.hideturtle()
animate_best_path.width(1.5)
animate_best_path.color("white")
animate_current_path = t.Turtle()
animate_current_path.hideturtle()
animate_current_path.color("grey")
list_points = [(609, 184), (-423, -335), (218, 47), (447, 56), (747, -124), (-377, -23), (215, 308), (286, -331), (76, -57), (358, 176), (135, 264), (34, -305), (552, -61), (733, 31), (-531, 343), (148, 180), (-454, -168), (695, -275), (548, -60), (-170, -302), (59, 156), (432, 210), (209, 333), (306, -192), (615, 68), (-539, 236), (-214, 50), (-300, 289), (-733, -313), (169, -165), (-514, -39), (50, 214), (-199, 55), (533, -205), (-669, 187), (741, 260), (668, 272), (-567, -145), (613, 141), (-722, 204), (172, -311), (303, -283), (450, -257), (423, 129), (69, 100), (-184, 98), (496, 230), (739, -160), (206, 209), (149, 137), (374, -159), (133, 113), (32, -228), (-434, 255), (63, -193), (-583, 97), (-32, -153), (-714, -153), (-580, -181), (415, 54), (-34, 208), (59, 306), (438, -111), (-387, -212), (-45, -52), (-360, 180), (-302, -2), (5, 1), (105, -91), (-484, -154), (-302, -179), (-468, -94), (378, 133), (-301, 1), (-658, -48), (-71, -13), (599, -80), (1, -146), (-616, 158), (-53, -269), (454, 247), (641, 268), (278, -105), (-369, 270), (739, 149), (388, 240), (-544, -257), (-346, 22), (-243, 159), (530, -228), (629, 161), (167, -207), (353, 1), (569, -304), (-113, 141), (-717, -206), (-176, -287), (-173, -49), (389, -248), (-231, 0)]
lock = False
edge_pheromones = {}
START_CONST = 1000
ALPHA = 2
BETA = 5
FADE_CONST = 0.8
PHEROMONE_CONST = 5000
best_path = []
best_dist = 1000000
path_found = []
list_distance = []
iteration = 0
def create_dictionary(list_points):
    #each key is (x1 + x2, y1 + y2)
    for i in list_points:
        for j in list_points:
            if (i == j):
                continue
            key = (i[0] + j[0], i[1] + j[1])
            if (key in edge_pheromones):
                continue
            edge_pheromones[key] = START_CONST
class Ant:
    @staticmethod
    def find_path(list_point, current_path, current_node):
        if (len(list_point) == 0):
            global path_found
            current_path.append(current_path[0]) #go back to starting point
            Ant.update_pheromones(current_path) #update pheromones and best path
            path_found = current_path
            return current_path
        edge_dist = {}
        top_five = {}
        for i in list_point:
            dist = m.sqrt(pow(current_node[0] - i[0], 2) + pow(current_node[1] - i[1], 2))
            # distance (key), node (value)
            edge_dist[dist] = i
        num_iterations = 0
        if (len(edge_dist) > 5):
            num_iterations = 5  
        else: 
            num_iterations = len(edge_dist)
        for i in range(0, num_iterations):
            #find the top five closest
            minimum_dist = min(edge_dist) #minimum distance
            node = edge_dist[minimum_dist] #node with minimum distance
            pheromones_key = (current_node[0] + node[0], current_node[1] + node[1])
            probability = pow(edge_pheromones[pheromones_key], ALPHA) / pow(minimum_dist, BETA)
            # node (key), probability (value)
            top_five[node] = probability
            edge_dist.pop(minimum_dist)
        final_node = Ant.biased_random(top_five) #returns the node
        list_point.remove(final_node)
        current_path.append(final_node)
        Ant.find_path(list_point, current_path, final_node)
    @staticmethod
    def update_pheromones(path):
        global best_dist, best_path, iteration, list_distance
        for i in edge_pheromones:
            if (edge_pheromones[i] > 0.001):
                edge_pheromones[i] *= FADE_CONST
        total_dist = Ant.path_length(path)
        if (total_dist < best_dist):
            global iteration
            best_path = path
            best_dist = total_dist
        list_distance.append(total_dist)
        pheromone_value = 3 * PHEROMONE_CONST / total_dist
        #calculate the total distance of the path
        for i in range(0, len(path) - 1):
            #recalculate the edge key to access the pheromone value in edge_pheromones
            node1 = path[i]
            node2 = path[i + 1]
            key = (node1[0] + node2[0], node1[1] + node2[1])
            edge_pheromones[key] += pheromone_value
    @staticmethod
    def path_length(path):
        total_distance = 0
        for i in range(0, len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]
            distance = m.sqrt(pow(node1[0] - node2[0], 2) + pow(node1[1] - node2[1], 2))
            total_distance += distance
        return total_distance
    @staticmethod 
    def biased_random(dict_probabilities):
        assert type(dict_probabilities) == dict
        list_weights = list(dict_probabilities.values())
        if (len(list_weights) == 1):
            return list(dict_probabilities.keys())[0]
        list_weights = sorted(list_weights)
        normalized_original = {} #normalized weight (key), original weight (value)
        try:
            normalizer = 1 / sum(list_weights)
        except (ZeroDivisionError):
            print(f"ERROR: {list_weights}")
            raise Exception("ERROR")
        for i in list_weights:
            #list_weights[i] *= normalizer
            normalized_original[i * normalizer] = i
        random = r.random()
        cumulative_frequency = 0
        chosen_weight = 0
        for i in range(0, len(list_weights)):
            weight = list(normalized_original)[i]
            cumulative_frequency += weight
            if (cumulative_frequency > random):
                chosen_weight = weight
                break
        return list(dict_probabilities.keys())[list(dict_probabilities.values()).index(normalized_original[chosen_weight])]
def draw_point():
    for i in list_points:
        t.goto(i)
        t.dot(5, "white")
        t.write(f"({i[0]}, {i[1]})", align = "center")
    t.update()
def click(x, y): 
    global lock
    if (lock):
        return
    list_points.append((m.floor(x), m.floor(y)))
    t.goto(x, y)
    t.dot(5, "white")
    t.write(list_points[len(list_points) - 1], align = "center")
    t.update()
def animate_pathfound():
    animate_current_path.clear()
    for i in range(0, len(path_found) - 1):
        animate_current_path.up()
        animate_current_path.goto(path_found[i])
        animate_current_path.down()
        animate_current_path.goto(path_found[i + 1])
    tscreen.update()
def animate_bestpath():
    animate_best_path.clear()
    for i in range(0, len(best_path) - 1):
        animate_best_path.up()
        animate_best_path.goto(best_path[i])
        animate_best_path.down()
        animate_best_path.goto(best_path[i + 1])
    tscreen.update()
def path_find():
    global iteration
    create_dictionary(list_points)
    for i in range(0, 1000):
        random_node = r.choice(list_points)
        list_point = c.deepcopy(list_points)
        list_point.remove(random_node)
        current_path = [random_node]
        Ant.find_path(list_point, current_path, random_node)
        animate_pathfound()
        animate_bestpath()
        if (i == 300 or i == 700 or i == 850):
            for i in edge_pheromones:
                edge_pheromones[i] = 0.001
            for i in range(0, len(best_path) - 1):
                point1 = best_path[i]
                point2 = best_path[i + 1]
                key = (point1[0] + point2[0], point1[1] + point2[1])
                edge_pheromones[key] = pow(PHEROMONE_CONST, 10)
        iteration += 1
    t.textinput("FINISHED", f"Best distance found: {int(best_dist)}")
    print(best_dist)
    mp.plot(list_distance)
    mp.title("Path Length over Time")
    mp.show()
def start():
    global lock
    if (len(list_points) > 2 and not lock):
        lock = True
        path_find()
draw_point()
tscreen.onkey(start, "space")
tscreen.listen()
#tscreen.onclick(click)
tscreen.mainloop()