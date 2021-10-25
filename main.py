# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:15:54 2021

@author: Alexandre
"""


import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage import data,filters, io
from skimage.transform import radon, rescale, iradon, rotate
import math as math;
import numpy as np
import imageio 
#obj = io.imread('disque.jpg')
obj = imageio.imread("image-reduite.png")
xlen = len(obj)
ylen = len(obj[0])
    

#!!! Les fonctions s'appliquent sur des matrices numpy (array) de dimension 2 !!!

#Convertit une image en couleur en une image en niveau de gris      
def gris(img):
    result = np.empty([xlen, ylen])
    for i in range(0, xlen):
        for j in range(0, ylen):
            avrg = abs(int(img[i,j,0]/3) + int(img[i,j,1]/3) + int(img[i,j,2]/3))
            result[i,j] = avrg
    return result;

#Image sous forme de niveau de gris sur laquel on peut maintenant travailler comme s'il c'était une "densité"
img_gris = gris(obj)

#Retourne la valeur de gris dans une case, si la case est en dehors de l'image le résultat est nul
def val(img, i,j):
    if i >= 0 and i < xlen and j >= 0 and j < ylen:
        return img[i,j];
    else:
        return 0;


#Change la couleur d'un pixel s'il est dans [0, xlen-1] * [0, ylen-1]
def colorize(img, i,j, c):
    if i >= 0 and i < xlen  and j >= 0 and j < ylen  :
        img[j,i] = c
   

k = round(math.sqrt(2),3)
     
#Affiche la projection caractérisée par theta et rho
def v_proj(img, theta, rho):
    tot = 0 
    N = 0
    for i in range(int(-xlen), int(xlen)):
        i,j = round(rho*math.cos(theta) - i*math.sin(theta)),round(rho*math.sin(theta) + i*math.cos(theta))
        colorize(img, i, j, 255)
    plt.figure(dpi=300)
    plt.imshow(img)
    plt.show()

#Effectue la somme des densités sur une ligne de projection (opération de projection)
def R__(obj, theta, rho):
    tot = 0 
    N = 1
    for i in range(int(-xlen), int(xlen)):
        i,j = round(rho*math.cos(theta) - i*math.sin(theta)),round(rho*math.sin(theta) + i*math.cos(theta))
        tot += val(obj, i,j)
    return int(tot);

def sinogram_bis(obj):
    projections = []
    M = 180
    for k in range(0, xlen):
        projections.append([])
        for m in range(0, M):
            projections[k].append(R__(obj, m * math.pi / M, k))
    
    plt.imshow(projections)
    plt.show()
    #plt.imshow(np.vstack(projections))
    #plt.show()
    
#Fonction trouvée dans un repo github
def sinogram_marche(image, steps):        
    projections = []
    dTheta = -180.0 / steps 
    
    for i in range(steps):
        plt.imshow(rotate(image, i*dTheta))
        projections.append(rotate(image, i*dTheta).sum(axis=0))

    
    final = np.vstack(projections)
    plt.imshow(projections)
    plt.show()
    plt.imshow(final)
    plt.show()
    imageio.imwrite("fig10.png", final)
    return final

def reverse(u):
    freq = math.pi * np.fft.fftfreq(xlen, xlen/2)
   # Q = np.fft.ifft(np.fft.fft(u))
                    
    return np.fft.ifft(np.fft.fft(u, axis = 0), axis = 0)
#v_proj(0, 1000)
    
