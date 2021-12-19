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
obj = imageio.imread("original.png")
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
    if i >= 0 and i < ylen and j >= 0 and j < xlen:
        return img[j,i];
    else:
        return 0;


#Change la couleur d'un pixel s'il est dans [0, xlen-1] * [0, ylen-1]
def colorize(img, i,j, c):
    if i >= 0 and i < xlen  and j >= 0 and j < ylen  :
        img[-j,i] = c
   

k = round(math.sqrt(2),3)
     
#Affiche la projection caractérisée par theta et rho
def v_proj(img, theta, rho):
    tot = 0 
    N = 0
    for k in range(-int(1.4*xlen), int(1.4*xlen)):
        i,j = round(rho*math.cos(theta) - k*math.sin(theta)),round(rho*math.sin(theta) + k*math.cos(theta))
        print(k, i,j)
        colorize(img, i, j, 255)

    plt.figure(dpi=300)
    plt.imshow(img)
    plt.show()
    
#Effectue la somme des densités sur une ligne de projection (opération de projection)
def R__(obj, theta, rho):
    tot = 0 
    N = 1
    for k in range(int(-1.4*xlen), int(1.4*xlen)):
        i,j = round(rho*math.cos(theta) - k*math.sin(theta) + xlen/2),round(rho*math.sin(theta) + k*math.cos(theta)+xlen/2)
        tot += val(obj, i,j)
    return int(tot);

def sinogram(obj):
    M = 180
    projections = np.zeros((M, obj.shape[0]))
    for k in range(-int(xlen/2), int(xlen/2)):
        for m in range(0, M):
            projections[m, -int(xlen/2)+k] = (R__(obj, m * math.pi / M, k))
    
    plt.imshow(projections)
    plt.show()
    #plt.imshow(np.vstack(projections))
    #plt.show()
    return rotate(projections,-180)
    
def value(tab, i, j):
    if i >= 0 and i < np.shape(tab)[0] and j >= 0 and j < np.shape(tab)[1]:
        return tab[i,j];
    else:
        return 0;


def reverse(im):
    xlen = im.shape[0]
    ylen = im.shape[1]
    resultat = np.zeros((ylen, ylen))
    dTheta = 180.0 / xlen
    for i in range(xlen):
        temp = np.tile(im[i], (ylen, 1))
        temp = rotate(temp, dTheta * i)
        resultat += temp
    return resultat

def Lambda(N, M):
    t = np.zeros((N, M))
    for k in range(N):
        t[k, :] = abs(math.pi * np.fft.fftfreq(M, 2 / N))
    return t

def Q(im, N, M):
    A = Lambda(N, M)  
    B = np.fft.fft(im, axis=1)  
    C = A * B  
    return np.fft.ifft(C, axis=1)  


def reverse2(im):
    xlen = im.shape[0]
    ylen = im.shape[1]
    im = Q(im, xlen, ylen).real  
    resultat = np.zeros((ylen, ylen))
    dTheta = 180.0 / xlen
    im = rotate(im, -90)
    for i in range(xlen):
        temp = np.tile(im[i], (ylen, 1))
        temp = rotate(temp, dTheta * i)
        resultat += temp
    return resultat
