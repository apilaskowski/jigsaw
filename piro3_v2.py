# -*- coding: utf-8 -*-
"""
Created on Thu May 26 10:30:42 2016

@author: Tomasz Sosnowski & Artur Laskowski
"""

from skimage.measure import find_contours, approximate_polygon
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import sys

def distancePointToPoint(p1,p2):
    return math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))

def distancePointToLine(A,B,C,p):
    return abs(A * p[1] + B * p[0] + C) / math.sqrt(A * A + B * B)

def createPoint(x, y):
    return [x, y]

def loadImage(path,i):
    filename=path+str(i)+'.png'
    img = cv2.imread(filename,-1)
    return img

def loadImages(path,N):
    images=[]
    for i in range(0,N):
        images.append(loadImage(path,i))
    return images

def calculateLineFactorA(a, b):
    return (b[0] - a[0]) / (b[1] - a[1])

def calculateLineFactorC(a, b):
    return ((a[1] * (b[0] - a[0])) / (b[1] - a[1])) * -1.0 + a[0]

def calculateLineFactors(contour):
    p1 , p2 = contour[0], contour[len(contour) - 1]
    return calculateLineFactorA(p1, p2), -1.0, calculateLineFactorC(p1, p2)

def findPointFarthestToLine(contour, facA, facB, facC):
    max, maxP = .0, a
    for point in contour:
        dist = distancePointToLine(facA, facB, facC, point)
        if dist > max:
            max, maxP = dist, p
    return maxP

def findTipPoint(contour):
    factorA, factorB, factorC = calculateLineFactors(contour)
    maxP = findPointFarthestToLine(contour, factorA, factorB, factorC)
    return maxP, A, C

def findTipPointCastedToLine(p1, p2, pTip, A, C):
    Cprim = pD[0] - pD[1]/A
    newPx = (Cprim - C) / (A - 1/A)
    newPy = A * newPx + C
    return createPoint(newPy, newPx)

def isBlob(picture, contour):
    tipPoint, A, C = findTipPoint(contour)
    castedPoint = findTipPointCastedToLine(contour[0], contour[len(contour) - 1], tipPoint, A, C)
    if picture[castedPoint[0], castedPoint[1], -1] != 0:
        return True
    return False

def calculateNearestSegmentsLenghtForPoint(contour,index):
    return distancePointToPoint(contour[index-1],contour[index]) \
        + distancePointToPoint(contour[index],contour[(index + 1) % len(contour)])

def calculateNearestSegmentsLenghtForList(contour, points):
    sum = .0
    for p in points:
        sum += calculateNearestSegmentsLenghtForPoint(contout, p)
    return sum

def containsPoint(a,p):
    for i in range(len(a)):
        if (a[i][0]==p[0] and a[i][1]==p[1]):
            return True
    return False

def calculateSumDistanceForList(apprx, points, i, j, k, o):
    suma = 0.0
    for p1 in points:
        for p2 in points:
            if p1 != p2:
                suma += distancePointToPoint(apprx[p1], apprx[p2])
    return suma

def findEdges(rgba):
    result=[]
    img1 = rgba[:,:,-1]
    img2 = np.zeros((img1.shape[0] + 20, img1.shape[1] + 20))
    img2[10:img1.shape[0] + 10, 10:img1.shape[1] + 10] = img1
    contours = find_contours(img2, 0)
    contour=contours[0]
    apprx=approximate_polygon(contours[0], tolerance=15.0)

    #print apprx
    maxpoints=[]

    #print(len(apprx))

    maxa, maxb, maxc, maxd, maxval = (0,0,0,0,0)
    for i in range(len(apprx)-1):
        for j in range(i+1, len(apprx)-1):
            for k in range(j+1, len(apprx)-1):
                for o in range(k+1, len(apprx)-1):
                    csum = getDistSum(apprx, i, j, k, o) + calculateNearestSegmentsLenghtForList(apprx, [i, j, k, o])
                    if csum > maxval:
                        maxval, maxa, maxb, maxc, maxd = csum, i, j, k, o
    maxpoints.append(apprx[maxa])
    maxpoints.append(apprx[maxb])
    maxpoints.append(apprx[maxc])
    maxpoints.append(apprx[maxd])

    characteristicSegments=[]
    apprx=apprx[1:]
    for i in range(len(maxpoints)):
        for j in range(len(apprx)):
            if ((apprx[j]==maxpoints[i-1]).all()):
                index1=j
            if ((apprx[j]==maxpoints[i]).all()):
                index2=j
        if ((index1+1)<len(apprx)):
            d1=distancePointToPoint(apprx[index1],apprx[index1+1])
        else:
            d1=distancePointToPoint(apprx[index1],apprx[0])
        d2=distancePointToPoint(apprx[index2-1],apprx[index2])
        if (d1==d2==distancePointToPoint(apprx[index1],apprx[index2])):
            d1=0.0
            d2=0.0
        #1+1)<len(apprx)):
        #    print "distances",apprx[index1],apprx[index1+1],d1,apprx[index2-1],apprx[index2],d2
        #else:
        #    print "distances",apprx[index1],apprx[0],d1,apprx[index2-1],apprx[index2],d2
        characteristicSegments.append([d1,d2])

    #print(apprx)
    #print(maxpoints)

    starting=0
    fstart=0
    #print maxpoints

    #plt.imshow(img2)
    #plt.plot(apprx[:,1],apprx[:,0])
    #maxp=np.asarray(maxpoints)
    #plt.plot(maxp[:,1],maxp[:,0], marker='o', color='r', ls='')
    #plt.show()

   # for i in range(len(contour)):
   #     colour=rgba[contour[i][0]-10,contour[i][1]-10,:]
   #     print contour[i], colour

    for i in range(4):
        tempc=[]
        for j in range(starting,2*len(contour)):
            ind=j%len(contour)
            #print maxpoints,contour[ind]
            tempc.append(np.array(contour[ind]).tolist())
            if (containsPoint(maxpoints,contour[ind]) and ind!=starting):
                #print maxpoints,contour[ind],ind,starting,tempc
                if (starting==0):
                    fstart=ind
                    tempc=[]
                starting=ind
                break
        if (len(tempc)>0):
            result.append(np.asarray(tempc))
    tempc=[]
    for i in range(starting,len(contour)):
        tempc.append(contour[i])
    for i in range(1,fstart):
        tempc.append(contour[i])
        result.append(np.asarray(tempc))
    #print 'Result:',result

    return result, characteristicSegments

def avgColour(edge,img):
    suma=[0,0,0]
    for i in range(len(edge)):
        for j in range(len(suma)):
            suma[j]+=img[edge[i][0]-10,edge[i][1]-10,j]
    result=[0.0,0.0,0.0]
    for i in range(len(suma)):
        result[i]=float(suma[i])/float(255*len(edge))
    return result

def compareAverageColour(edge1,edge2,img1,img2):
    col1=avgColour(edge1,img1)
    col2=avgColour(edge2,img2)
    suma=0.0
    for i in range(3):
        suma+=abs(col1[i]-col2[i])
    suma=suma/3.0
    return suma

def compareShape(edges1,edges2):
    #img3=transformImage(img1)
    #img4=transformImage(img2)
    #contours1=getContour(img1)
    #contours2=getContour(img2)
    #contours1=find_contours(img1[:,:,-1],0)
    #contours2=find_contours(img2,0)
    #edges1=findEdges(img1)
    #edges2=findEdges(img2)
    mincomp=100.0
    for edge1 in edges1:
        for edge2 in edges2:
            #print edge1
            comp=cv2.matchShapes(edge1,edge2,1,0.0)
            #print comp, edge1[0],edge1[-1], edge2[0],edge2[-1]
            #coloursComp=compareAverageColour(edge1,edge2,img1,img2)
            #comp2=comp+3.0*coloursComp
            if (comp<mincomp and len(edge1)>5 and len(edge2)>5):
                mincomp=comp
            #break
        #break
    #print edges1[0]
    #cnt1=contours1[0]
    #print cnt1
    #cnt2=contours2[0]
    return mincomp
    #return cv2.matchShapes(cnt1,cnt2,1,0.0)

def compareByProportions(seg1,seg2):
    mincomp=100.0
    for i in range(len(seg1)):
        for j in range(len(seg2)):
            if (seg1[i][0]==0.0 or seg1[i][1]==0.0 or seg2[j][0]==0.0 or seg2[j][1]==0.0):
                comp=200.0
            else:
                d1=seg1[i][0]/seg1[i][1]
                d2=seg2[j][0]/seg2[j][1]
                print d1,d2
                if ((d1>1.0 and d2<1.0) or (d1<1.0 and d2>1.0)):
                    d1=1.0/d1
                #d1=seg1[i][1]/seg2[j][0]
                #d2=seg1[i][0]/seg2[j][1]
                comp=abs((d1/d2)-1.0)
            if (comp<mincomp):
                mincomp=comp
    return mincomp

def calculateEdgeHistogram(edge):
    a=edge[0]
    b=edge[-1]
    A = calcA(a, b)
    B = -1.0
    C = calcC(a, b)
    d=distance(a,b)
    hist=np.zeros(100)
    #print a,b,d
    for i in range(len(edge)):
        ratio=0.0
        if (a[1]==b[1]):
            ratio=abs(a[1]-edge[i][1])/d
        else:
            ratio=calculateDistance(A,B,C,edge[i])/d
        #print A,B,C,edge[i],calculateDistance(A,B,C,edge[i])
        index=int(ratio*100.0)
        hist[index]+=(1.0/float(len(edge)))
    return hist

def compareHistograms(hist1,hist2):
    error=0.0
    for i in range(len(hist1)):
        error+=((hist1[i]-hist2[i])**2)
    return error

def compareDistances(shape1,shape2):
    mincomp=10000.0
    for edge1 in shape1:
        hist1=calculateEdgeHistogram(edge1)
        for edge2 in shape2:
            hist2=calculateEdgeHistogram(edge2)
            comp=compareHistograms(hist1,hist2)
            if (comp<mincomp and len(edge1)>5 and len(edge2)>5):
                mincomp=comp
    return mincomp

def colorHist(edge,img,channel):
    counter=0
    histtemp=np.zeros(100)
    for i in range(len(edge)/10):
        j=i*10
        y=edge[j][0]-10
        x=edge[j][1]-10
        square=img[max(0,y-5):min(y+5,img.shape[0]),max(0,x-5):min(x+5,img.shape[1]),:]
        for y1 in range(square.shape[0]):
            for x1 in range(square.shape[1]):
                value=square[y1,x1,:]
                if (value[-1]>0):
                    counter+=1
                    index=int(float(value[channel])/float(len(histtemp)))
                    histtemp[index]+=1
    hist=np.zeros(100)
    for i in range(len(hist)):
        hist[i]=float(histtemp[i])/float(counter)
    return hist


def compareColours(shape1,shape2,img1,img2):
    mincomp=10000.0
    for edge1 in shape1:
        histr1=colorHist(edge1,img1,0)
        histg1=colorHist(edge1,img1,1)
        histb1=colorHist(edge1,img1,2)
        for edge2 in shape2:
            histr2=colorHist(edge2,img2,0)
            histg2=colorHist(edge2,img2,1)
            histb2=colorHist(edge2,img2,2)
            comp=compareHistograms(histr1,histr2)+compareHistograms(histg1,histg2)+compareHistograms(histb1,histb2)
            if (comp<mincomp and len(edge1)>5 and len(edge2)>5):
                mincomp=comp
    return mincomp

def mostDistantRatio(edge):
    maxratio=0.0
    a=edge[0]
    b=edge[-1]
    A = calcA(a, b)
    B = -1.0
    C = calcC(a, b)
    d=distance(a,b)
    for i in range(len(edge)):
        ratio=calculateDistance(A,B,C,edge[i])/d
        if (ratio>maxratio):
            maxratio=ratio
    return maxratio

def compareDistantPoint(shape1,shape2):
    mincomp=10000.0
    for edge1 in shape1:
        d1=mostDistantRatio(edge1)
        for edge2 in shape2:
            d2=mostDistantRatio(edge2)
            comp=20000.0
            if (d1>=d2 and d2!=0.0):
                comp=abs((d1/d2)-1.0)
            if (d2>=d1 and d1!=0.0):
                comp=abs((d2/d1)-1.0)
            if (comp<mincomp and len(edge1)>5 and len(edge2)>5):
                mincomp=comp
    return mincomp

def calculateNeighbours(shape):
    counter=0
    for edge in shape:
        if (len(edge)>10):
            counter+=1
    return counter

def printResult(best,shape):
    number=calculateNeighbours(shape)
    for i in range(number):
        print best[i],
    sys.stdout.write('%c'%'\t')
    for i in range(number,len(best)):
        print best[i],
    sys.stdout.write('%c'%'\n')

def main():
    path=sys.argv[1]
    N=int(sys.argv[2])
    images=(path,N)
    shapes=[]
    seg=[]
    for i in range(len(images)):
        edges,segments=findEdges(images[i])
        shapes.append(edges)
        seg.append(segments)
        #shapes.append(findEdges(images[i]))
    for i in range(len(shapes)):
        compares=np.ones(len(shapes))
        #print compares
        for j in range(len(shapes)):
            shape1=shapes[i]
            shape2=shapes[j]
            #compareShape(img1,img2)
            #print "Compare",i,j, compareByProportions(seg[i],seg[j])
            #compares[j]=compareShape(shape1,shape2)
            #compares[j]=compareByProportions(seg[i],seg[j])
            compares[j]=compareColours(shape1,shape2,images[i],images[j])+compareShape(shape1,shape2)+compareDistantPoint(shape1,shape2)+compareDistances(shape1,shape2)
            #compares[j]=compareDistances(shape1,shape2)
            #print "Compare",i,j,compares[j]
            #print i,j,compareShape(img1,img2)
            #plt.imshow(img1)
            #plt.show()
            #break
        compares[i]=100.0
        best=np.argsort(compares)
        printResult(best,shapes[i])
        #break
    descriptions=[]

main()
