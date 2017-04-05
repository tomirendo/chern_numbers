get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
import numpy as np
import pandas as pd
from collections import namedtuple

index = namedtuple("index", "phi, theta")

def mat(theta, phi):
    return np.matrix([[np.cos(theta), np.sin(theta)*np.exp(-1j*phi)],
                      [np.sin(theta)*np.exp(1j*phi), -np.cos(theta)]])

def normalized_dot_product(a,b):
    prod = np.vdot(a.tolist()[0],b.tolist()[0])
    return prod/np.abs(prod)

def is_close(a,b):
    #this is not well defined
    return (a-b)/max([a,b]) < .5

class Point:
    def __init__(self, index, delta_phi, delta_theta, mat):
        self.delta_phi, self.delta_theta = delta_phi, delta_theta
        self.phi, self.theta= (index.phi)*delta_phi, (index.theta)*delta_theta
        theta, phi = self.theta, self.phi
        self.x,self.y,self.z = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
        self.mat = mat
        self.matrix = mat(theta = self.theta, phi = self.phi)
        self.eig = np.linalg.eigh(self.matrix)
        self.vals = self.eig[0]
        self.vecs = self.eig[1].T
        self.index = index
        
    def __repr__(self):
        return "<θ={},φ={}>".format(self.theta, self.phi)


class Vertex:
    def __init__(self, points = None, sub_divide = True):
        self.points = points
        self.n = len(points)
        self.sub_vertexes = None
        if sub_divide:
            self.sub_divide()
        else : 
            pass

        
    def sub_divide(self):
        if self.n == 4 :
            mat, delta_theta, delta_phi= self.points[0].mat, self.points[0].delta_theta, self.points[0].delta_phi
            p1,p2,p3,p4 = self.points
            #Drawing two rectangles using P1...P4:
            # |P1 P5| |P5 P4|
            # |P2 P6| |P6 P3|
            p5_index = index(theta = p1.index.theta, 
                             phi = p1.index.phi*2+1)
            p6_index = index(theta = p2.index.theta, 
                             phi = p2.index.phi*2+1)
            p5 = Point(p5_index, delta_phi/2, delta_theta, mat)
            p6 = Point(p5_index, delta_phi/2, delta_theta, mat)
            self.sub_vertexes = [Vertex([p1,p2,p6,p5], sub_divide = False), Vertex([p5,p6,p3,p4], sub_divide = False)]
        elif self.n == 3:
            pass
        else :
            raise Exception("Vertex with n = {}".format(self.n))

    def calc(self, j):
        approximation = self.approximate(j)
        if self.sub_vertexes is not None:
            v1, v2 = self.sub_vertexes
            if is_close(v1.approximate(j), v2.approximate(j)):
                return approximation
            else :
                v1.sub_divide()
                v2.sub_divide()
                return v1.calc(j) + v2.calc(j)

    def approximate(self, j):
        res = 1
        for i in range(self.n):
            vec_1 = self.points[i].vecs[j]
            vec_2 = self.points[(i+1)%self.n].vecs[j]
            res *= normalized_dot_product(vec_1,vec_2)
        return np.log(res)
  
    def plot(self,ax = None):
        if ax is None:
            ax = Axes3D(plt.figure())
        for color, point in zip("rgby",self.points):
            ax.scatter3D([point.x],[point.y],[point.z], s=50, c=color)
            
    def __repr__(self):
        return "<{},{}>".format(self.points[0].index, str.join("\n", map(repr,self.points)))

class Sphere(dict):
    def __init__(self, delta_phi, delta_theta, mat = mat, length = 2):
        n_theta = int((np.pi ) // delta_theta) + 1
        n_phi = int((2*np.pi ) // delta_phi )
        dict.__init__(self)
        self.n_phi, self.n_theta = n_phi, n_theta
        self.delta_theta = delta_theta
        self.delta_phi = delta_phi
        self.verteces = []
        self.mat = mat
        self.length = length
        self.init_points()
        
    def init_points(self):
        n_phi, n_theta, delta_phi, delta_theta= self.n_phi, self.n_theta, self.delta_phi, self.delta_theta
        for phi, theta in product(range(0,n_phi),range(1,n_theta)):
            idx = index(phi = phi, theta = theta)
            p = Point(index=idx, delta_phi=delta_phi, delta_theta=delta_theta, mat=self.mat)
            self[idx] = p
            
        self.find_verteces()
        
        upper_pole_index = index(theta = 0, phi = 0)
        upper_pole = Point(upper_pole_index, delta_theta=delta_theta, delta_phi=delta_phi,mat = self.mat)
        self[upper_pole_index] = upper_pole
        
        for phi in range(1,n_phi):
            idx_2 = index(theta = 1, phi = phi)
            idx_3= index(theta = 1, phi = (phi + 1)%n_phi)
            self.verteces.append(Vertex(points = [upper_pole, self[idx_2], self[idx_3]]))
        
    def plot(self):
        xs,ys,zs = zip(*[[i.x,i.y,i.z] for i in self.values()])
        ax = Axes3D(plt.figure())
        ax.scatter3D(xs, ys, zs)
        
    def find_verteces(self):
        sm = 0
        for index in self:
            try : 
                self.verteces.append(Vertex(points=self.square(point = self[index])))
            except KeyError as e:
                pass
    def square(self, point):
        idx = point.index
        p2_index = index(phi = (idx.phi), theta = (idx.theta+1))
        p3_index = index(phi = (idx.phi + 1)%self.n_phi, theta = (idx.theta + 1))
        p4_index = index(phi = (idx.phi + 1)%self.n_phi , theta = (idx.theta))
        return [point, self[p2_index], self[p3_index], self[p4_index]]

    @property
    def df(self):
        length = self.length
        columns = ['f{}'.format(i) for i in range(length)]
        df = pd.DataFrame([[i.calc(j) for j in range(length)] for i in self.verteces], columns = columns)
        for col in columns:
            df[col.upper()] = df[col].imag
        return df

    def chern_numbers(self):
        df = self.df
        return [int(np.round(df['F{}'.format(i)].sum()/(2*np.pi))) for i in range(self.length)]




# 
