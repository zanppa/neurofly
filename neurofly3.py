# -*- coding: utf-8 -*-
"""
Created on Fri Aug 04 22:07:05 2017

@author: Zan

Original source from https://iamtrask.github.io/2015/07/12/basic-python-network/

"""

import numpy as np

import matplotlib.pyplot as plt

plt.ion()
plt.show()


def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))


# Always the same seed to make debug easier    
np.random.seed(1)

generations = 701

speed = 0.1
steps = 300
wall_left = -0.5
wall_right = 0.5

spheres = np.int(np.ceil(speed * steps / 5.0))
spherex = []
spherey = []
sphererad = []
sp_ave_size = 0.5
sp_ave_dist = 20.0

def generate_spheres():
    spherex[:] = []
    spherey[:] = []
    sphererad[:] = []
    
    # Generate sphere locations
    for n in range(spheres/2):
        spherex.append(wall_left + 0.2 * np.random.random_sample() - 0.1)
        spherey.append((n+0.5) * sp_ave_dist + 2.0 * np.random.random_sample() - 1.0)
        sphererad.append(sp_ave_size + 0.2*np.random.random_sample()-0.1)

        spherex.append(wall_right + 0.2 * np.random.random_sample() - 0.1)
        spherey.append((n+1.0) * sp_ave_dist + 2.0 * np.random.random_sample() - 1.0)
        sphererad.append(sp_ave_size + 0.2*np.random.random_sample()-0.1)    


def circle_line_dist(x, y, angle, cx, cy, rad):
    """ Find distance from vector from (x,y) and angle in radto circle in cx, cy with radius rad """
    # Line-of-sight vector
    dx = np.sin(angle)
    dy = np.cos(angle)

    # Left normal for the vector
    lnx = -dy
    lny = dx
    
    # Vector from point to circle center
    cdx = cx - x
    cdy = cy - y

    # Dot product -> length from center to line-of-sight
    dot = abs(lnx*cdx + lny*cdy)
    
    # Collision if dot is < radius
    if dot <= rad:
        # Return distance to circle edge
        return np.sqrt(cdx**2 + cdy**2) - rad
    
    # Otherwise return -1.0 (negative indicates no collision)
    return -1.0
    

def sense_world(x, y, angle):
    sensor1 = np.deg2rad(angle + 40)
    sensor2 = np.deg2rad(angle - 40)

    # Sense walls
    # Clamp minimum to zero because those are "behind"
    sense = np.array([0.0, 0.0])
    sense[0] = np.sin(sensor1) / (wall_right - x)
    sense[0] = max(0.0, sense[0], np.sin(sensor1) / (wall_left - x))
    
    sense[1] = np.sin(sensor2) / (wall_left - x)
    sense[1] = max(0.0, sense[1], np.sin(sensor2) / wall_right - x)

    # Sense circles
    for i in range(len(spherex)):
        dist = circle_line_dist(x, y, sensor1, spherex[i], spherey[i], sphererad[i])
        if dist > 0.0:
            sense[0] = max(sense[0], 1.0/dist)

        dist = circle_line_dist(x, y, sensor2, spherex[i], spherey[i], sphererad[i])
        if dist > 0.0:
            sense[1] = max(sense[0], 1.0/dist)


    # Scale and limit sense range
    sense *= 0.2
    sense = np.clip(sense, 0.0, 1.0)
    
    return sense[:]


def collision(x, y):
    # Check walls
    if x <= wall_left or x >= wall_right:
        return True

    # Check circles
    for i in range(len(spherex)):
        dist2 = (spherex[i] - x)**2 + (spherey[i] - y)**2
        if dist2 <= sphererad[i]**2:
            return True

    return False


def draw_world(ax):
    # Draw world
    ax.plot([0, speed*steps], [wall_left, wall_left])
    ax.plot([0, speed*steps], [wall_right, wall_right])
    
    # Draw spheres
    for i in range(len(spherex)):
        ax.add_artist(plt.Circle((spherey[i], spherex[i]), sphererad[i]))

    return
        
        
        
class Brain:
    def __init__(self):
        # Network is :
        # 2 inputs
        # 4 hidden nodes, all inputs connected to all nodes
        # 2 outputs, all hidden nodes connected to both
        
        self.new_brain()

        self.l0 = [0.0]
        self.l1 = [0.0]
        self.l2 = [0.0]
        
        return
    
    def new_brain(self):
        # randomly initialize our weights with mean 0
        # First layer has 2 inputs and 4 synapses
        self.syn0 = 2*np.random.random((2,4)) - 1
        
        # Second layer has 4 inputs and 2 synapses = output
        self.syn1 = 2*np.random.random((4,2)) - 1
        return
    
    def calc(self, X):
        """ Feed forward through neural layers """
        self.l0 = X
        self.l1 = nonlin(np.dot(self.l0,self.syn0))
        self.l2 = nonlin(np.dot(self.l1,self.syn1))
        
        return self.l2


    def clone(self, other):
        """ Copy other brain to this one """
        self.syn0 = np.copy(other.syn0)
        self.syn1 = np.copy(other.syn1)
        #np.copyto(self.syn0, other.syn0)
        #np.copyto(self.syn1, other.syn1)
        return


    def evolve(self, amount):
        """ Tune the weights randomly """
        for element in np.nditer(self.syn0, op_flags=['readwrite']):
            element += amount * np.random.random_sample() - (amount / 2.0)

        for element in np.nditer(self.syn1, op_flags=['readwrite']):
            element += amount * np.random.random_sample() - (amount / 2.0)

        return




# Create brain
class Fly:
    def __init__(self):
        self.br = Brain()
        self.cx = [0.0]
        self.cy = [0.0]

        self.plaa = [0.0]
        self.s1 = [0.0]
        self.s2 = [0.0]

        self.reset()
        
        return

    def calc(self):
        if collision(self.cx[-1], self.cy[-1]):
            self.alive = False
            return

        sense = sense_world(self.cx[-1], self.cy[-1], self.ang)
    
        #print sense[0], sense[1]
        self.s1.append(sense[0])
        self.s2.append(sense[1])
    
        react = self.br.calc(sense)
        
        

        # If output is 1 0 turn left or 0 1 turn right, otherwise go straight
        if react[0] > 0.5 and react[1] < 0.5:
            self.ang -= 1.5
            self.plaa.append(-1.0)
        elif react[0] < 0.5 and react[1] > 0.5:
            self.ang += 1.5
            self.plaa.append(1.0)
        else:
            self.plaa.append(0.0)

        xnew = self.cx[-1] + speed * np.sin(np.deg2rad(self.ang))
        ynew = self.cy[-1] + speed * np.cos(np.deg2rad(self.ang))
        self.cx.append(xnew)
        self.cy.append(ynew)
    
        if self.cy[-1] > self.distance:
            self.distance = self.cy[-1]

    def clone(self, other):
        """ Clone the brain of the fly """
        self.br.clone(other.br)
        return
        
    def evolve(self, amount):
        self.br.evolve(amount)
        return
        
    def reset(self):
        # Initialize position and everything
        self.cx[:] = [0.0]
        self.cy[:] = [0.0]
        self.ang = 10.0*np.random.random_sample()-5.0
        self.distance = 0.0
        self.alive = True
        
        self.plaa[:] = [0.0]
        self.s1[:] = [0.0]
        self.s2[:] = [0.0]

        return
        
    def reset_brain(self):
        self.br.new_brain()
        return

    def __lt__(self, other):
         """ Compare which instance is better than other """
         # Inverse, i.e. "longer is better (smaller)"
         return self.distance > other.distance


# First do the evolution steps
plt.figure(1)

# Generate blocks randomly
generate_spheres()


# Create 25 flys
flys = []
for i in range(25):
    flys.append(Fly())

# Evolve
for evostep in range(generations+1):

    # Calculate life for maximum of 500 cycles
    for flyz in flys:
        for i in range(steps):
            flyz.calc()
            if not flyz.alive:
                break
        
    # Arrange flys so that farthest flyers are first
    flys.sort()
    
    # Plot results every now and then
    if evostep % 100 == 0:
        print "Step: ", evostep
        print "Best distance:", flys[0].distance, "max", speed*steps

        fig = plt.figure(1)
        plt.clf()
        #axes = plt.gca()
        ax = fig.add_subplot(211)
        #ax.cla()
        draw_world(ax)
        
        for (n,flyz) in enumerate(flys):
            if n == 0 or flyz.alive:
                ax.plot(flyz.cy, flyz.cx)

        ax2 = fig.add_subplot(212)
        ax2.plot(flys[0].plaa)
        ax2.plot(flys[0].s1)
        ax2.plot(flys[0].s2)

        #plt.draw()
        plt.pause(0.001)


    # If five best got long enough, increase distance and re-generate
    if flys[4].distance > speed*steps*0.9:
        steps += 50

        # Generate blocks randomly
        spheres = np.int(np.ceil(speed * steps / 5.0))
        generate_spheres()
        
        

    # Reset all flys
    for flyz in flys:
        # Reset flys position (does not reset the brain)
        flyz.reset()


    # If 5 best ones get to the end -> end simulation
#    if flys[4].distance > speed * (steps - steps/20.0) or evostep == generations:
#        print "Evolution finished in", evostep, "steps"
#        break

    # Select 5 best, clone them 4 times and evolve 3 of them
    # i.e. 0-20 are clones of 10 best
    # 5-19 are evolved clones of those
    # 20-25 are new random ones
    # Randomize others
    clone = 5
    amount = 4
    for n in range(1, amount):
        for i in range(clone):
            flys[n*clone + i].clone(flys[i])

    # Evolve
    for n in range(5, 20):
        flys[n].evolve(0.4)

    # Randomize rest
    for n in range(20, 25):
        flys[n].reset_brain()



print "Doing final test..."

steps = 10000
spheres = np.int(np.ceil(speed * steps / 10.0))
generate_spheres()

plt.figure(2)

# After evolution, run 5 best ones for looong time
for flyz in flys:
    for i in range(steps):
        flyz.calc()
        if not flyz.alive:
            break

# And plot the results
plt.clf()
ax = plt.gca()
ax.cla()
draw_world(ax)
for flyz in flys:
    ax.plot(flyz.cy, flyz.cx)

