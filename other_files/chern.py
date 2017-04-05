
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import *
import pandas as pd


def normalized_dot_product(a,b):
    prod = np.vdot(a.tolist()[0],b.tolist()[0])
    return prod/np.abs(prod)

ROUNDING_DIGITS = 5
problematic_vertexes = {378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 421}
class Point:
    def __init__(self, string = None, theta = None, phi = None, mat = None):
        if string is not None:
            x,y,z = map(float,string.split()[:3])
            self.x, self.y, self.z  = x,y,z 
            self.theta = round(np.arccos(z/(x**2+y**2+z**2)), ROUNDING_DIGITS)
            self.phi = round(np.arctan2(x,y), ROUNDING_DIGITS)
            if self.phi < -pi*.8:
                self.phi += 2*pi
        else :
            self.theta, self.phi = round(theta,ROUNDING_DIGITS), round(phi,ROUNDING_DIGITS)
            x,y,z = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
            self.x, self.y, self.z  = x,y,z 

        self.matrix = mat(self.theta, self.phi)
        self.eig = np.linalg.eigh(self.matrix)
        self.vals = self.eig[0]
        self.vecs = self.eig[1].T
    def __repr__(self):
        return "<θ={},φ={}>".format(self.theta, self.phi)

class Vertex:
    def __init__(self, string = None, points = None):
        if string is not None:
            #Assumes globals points
            n = int(string.split()[0])
            idx = map(int, string.split()[1:1+n])
            self.points = [points[i] for i in idx]
            self.n = n 
            self.sort()
        else :
            self.points = points
            self.n = len(points)
    def calc(self, j):
        res = 1
        for i in range(self.n):
            vec_1 = self.points[i].vecs[j]
            vec_2 = self.points[(i+1)%self.n].vecs[j]
            res *= normalized_dot_product(vec_1,vec_2)
        return np.log(res)
    def plot(self,ax = None):
        if ax is None:
            fg = plt.figure()
            ax = Axes3D(fg)
        for color, point in zip("rgby",self.points):
            ax.scatter3D([point.x],[point.y],[point.z], s=50, c=color)
    def sort(self):
        if self.n != 4: 
            self.points = self.points[::-1]
            return
        self.points.sort(key = lambda x:(-x.theta, x.phi))
        minimum, a, b, maximum = self.points
        if a.phi > b.phi:
            self.points = [minimum, a,maximum, b]
        else :
            self.points = [minimum, b,maximum, a]
        
    def __repr__(self):
        return "<{}>".format(str.join("\n", map(repr,self.points)))

class Sphere(list):
    def plot(self):
        xs,ys,zs = zip(*[[i.x,i.y,i.z] for i in self])
        ax = Axes3D(plt.figure())
        ax.scatter3D(xs, ys, zs)
    def get_point(self, theta, phi):
        theta = round(theta, ROUNDING_DIGITS)
        pih  = round(phi, ROUNDING_DIGITS)
        for i in self:
            if all(np.isclose((theta, phi), (i.theta, i.phi))):
                return i 
        raise Exception("Cannot find point {},{}".format(theta, phi))

