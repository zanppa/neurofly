# -*- coding: utf-8 -*-
"""
Created on Fri Aug 04 22:07:05 2017

@author: Zan

Original source from https://iamtrask.github.io/2015/07/12/basic-python-network/

"""

import numpy as np

import matplotlib.pyplot as plt
from math import sin, cos, radians
from time import sleep

plt.ion()
plt.show()


def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))


# Always the same seed to make debug easier    
np.random.seed(1)

wall_left = -1.0
wall_right = 1.0

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
        
        # Second layer has 4 inputs and 1 synapse = output
        self.syn1 = 2*np.random.random((4,1)) - 1
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
        self.reset()
        return

    def calc(self):
        if self.cx[-1] <= -1.0 or self.cx[-1] >= 1.0:
            self.alive = False
            return

        # Calculate distance to wall
        sense = np.array([0.0, 0.0])
        sense[0] = cos(radians(50 - self.ang)) / (wall_right - self.cx[-1])
        sense[1] = -cos(radians(-self.ang - 50)) / (wall_left - self.cx[-1])
    
        # Scale and limit sense range
        sense *= 0.3
        sense = np.clip(sense, 0.0, 1.0)
    
        react = self.br.calc(sense)

        # Control mode 1
        if react < 0.35:
            self.ang -= 1.5
        elif react > 0.65:
            self.ang += 1.5

        # Control mode 2
        #self.ang += 2.0*react - 1.0

        #self.ang = self.ang % 360

        xnew = self.cx[-1] + 0.1 * np.sin(np.deg2rad(self.ang))
        ynew = self.cy[-1] + 0.1 * np.cos(np.deg2rad(self.ang))
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
        self.ang = 60.0*np.random.random_sample()-30.0
        self.distance = 0.0
        self.alive = True

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

# Create 25 flys
flys = []
for i in range(25):
    flys.append(Fly())

# Evolve
for evostep in range(1001):

    # Calculate life for maximum of 500 cycles
    for flyz in flys:
        for i in range(500):
            flyz.calc()
            if not flyz.alive:
                break
        
    # Arrange flys so that farthest flyers are first
    flys.sort()
    
    # Plot results every now and then
    if evostep % 100 == 0:
        print "Step: ", evostep
        print "Best distance:", flys[0].distance
        plt.clf()
        for (n,flyz) in enumerate(flys):
            if n == 0 or flyz.alive:
                plt.plot(flyz.cy, flyz.cx)

        #plt.draw()
        plt.pause(0.001)

    # Reset all flys
    for flyz in flys:
        # Reset flys position (does not reset the brain)
        flyz.reset()

    
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
        flys[n].evolve(0.2)

    # Randomize rest
    for n in range(20, 25):
        flys[n].reset_brain()



print "Doing final test..."

plt.figure(2)

# After evolution, run 5 best ones for looong time
for flyz in flys[0:5]:
    for i in range(10000):
        flyz.calc()
        if not flyz.alive:
            break

# And plot the results
plt.clf()
for flyz in flys[0:5]:
    plt.plot(flyz.cy, flyz.cx)

