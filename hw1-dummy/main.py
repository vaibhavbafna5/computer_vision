"""
Starter code for EECS 442 W20 HW1
"""
from util import *
import numpy as np
import matplotlib.pyplot as plt
import skimage


def rotX(theta):
    # TODO: Return the rotation matrix for angle theta around X axis
    return np.array([[1, 0, 0], [0, np.cos(theta), -1 * np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])


def rotY(theta):
    # TODO: Return the rotation matrix for angle theta around Y axis
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-1 * np.sin(theta), 0, np.cos(theta)]])



def part1():
    inputs = []
    for i in range(40):
        inputs.append(rotY(i * np.pi / 20))
    generate_gif(inputs, 'cube.gif')
    
    renderCube(R=rotY(np.pi/4).dot(rotX(np.pi/4)), file_name="cube1")
    renderCube(R=rotX(np.pi/4).dot(rotY(np.pi/4)), file_name="cube2")

    # 1c
    renderCube(R=rotX(201*np.pi/1024).dot(rotY(np.pi/4)), file_name='1c')
    
    # 1d
    renderCube(f=np.inf, R=rotX(201*np.pi/1024).dot(rotY(np.pi/4)), file_name='1d')

   
def split_triptych(trip):
    # TODO: Split a triptych into thirds and return three channels in numpy arrays
    origPic = imageio.imread(trip[0])
    sliceSize = trip[1][1]
    startIndex = trip[1][0]
    stacker = []
    for i in range(3):
        picSlice = origPic[startIndex:startIndex+sliceSize, trip[1][2]:trip[1][3]]
        stacker.insert(0, picSlice)
        startIndex += sliceSize
    stacked = np.dstack(stacker)
    file = trip[0].split('/')
    file = file[1].split('.')
    imageio.imsave(file[0]+'_combined.jpg', stacked)
    return stacked

def normalized_cross_correlation(ch1, ch2):
    # TODO: Implement the default similarity metric
    ch1 = ch1 - ch1.mean(axis=0)
    ch2 = ch2 - ch2.mean(axis=0)
    return np.sum(np.divide(ch1, np.linalg.norm(ch1)) * np.divide(ch2, np.linalg.norm(ch2)))


def best_offset(ch1, ch2, metric, Xrange=np.arange(-15, 16), Yrange=np.arange(-15, 16)):
    # TODO: Use metric to align ch2 to ch1 and return optimal offsets
    minimum = -1
    for i in Xrange:
        for j in Yrange:
            sim = metric(ch1, np.roll(ch2,[i,j], axis=(0,1)))
            if sim > minimum:
                minimum = sim
                output = [i,j]
    print(output)
    return output


def align_and_combine(R, G, B, metric):
    # TODO: Use metric to align three channels and return the combined RGB image
    print("Blue to Red:")
    BtoR = best_offset(R, B, metric)
    print("Green to Red:")
    GtoR = best_offset(R, G, metric)
    B = np.roll(B, BtoR, axis=(0,1))
    G = np.roll(G, GtoR, axis=(0,1))
    colored = np.dstack([R,G,B])
    return colored


def part2():
    images = ['prokudin-gorskii/00125v.jpg','prokudin-gorskii/00149v.jpg',
              'prokudin-gorskii/00153v.jpg','prokudin-gorskii/00351v.jpg',
              'prokudin-gorskii/00398v.jpg','prokudin-gorskii/01112v.jpg',
              'tableau/efros_tableau.jpg']
    sliceInfo = [[26,330,25,393],[12,334,26,388],[13,334,25,382],
                 [15,332,24,376],[25,330,15,377],[15,330,14,376],
                 [0,420,0,507]]
    parser = [images,sliceInfo]
    for i in range(len(parser[0])):
        img = split_triptych([parser[0][i],parser[1][i]])
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        print(images[i] + " optimal offset")
        colored = align_and_combine(r,g,b,normalized_cross_correlation)
        file = parser[0][i].split('/')
        file = file[1].split('.')
        imageio.imsave(file[0] + '_finished.jpg', colored)


def part3():
    indoorPic = imageio.imread('rubik/indoor.png')
    outdoorPic = imageio.imread('rubik/outdoor.png')
    f, axarr = plt.subplots(4,3)
    axarr[0][0].imshow(indoorPic[:,:,0],cmap='gray',vmin=0,vmax=255)
    axarr[0][1].imshow(indoorPic[:,:,1],cmap='gray',vmin=0,vmax=255)
    axarr[0][2].imshow(indoorPic[:,:,2],cmap='gray',vmin=0,vmax=255)
    axarr[1][0].imshow(outdoorPic[:,:,0],cmap='gray',vmin=0,vmax=255)
    axarr[1][1].imshow(outdoorPic[:,:,1],cmap='gray',vmin=0,vmax=255)
    axarr[1][2].imshow(outdoorPic[:,:,2],cmap='gray',vmin=0,vmax=255)
    inConverted = skimage.color.rgb2lab(indoorPic[:,:,:3])
    outConverted = skimage.color.rgb2lab(outdoorPic[:,:,:3])
    axarr[2][0].imshow(inConverted[:,:,0],cmap='gray',vmin=0,vmax=100)
    axarr[2][1].imshow(inConverted[:,:,1],cmap='gray',vmin=-128,vmax=127)
    axarr[2][2].imshow(inConverted[:,:,2],cmap='gray',vmin=-128,vmax=127)
    axarr[3][0].imshow(outConverted[:,:,0],cmap='gray',vmin=0,vmax=100)
    axarr[3][1].imshow(outConverted[:,:,1],cmap='gray',vmin=-128,vmax=127)
    axarr[3][2].imshow(outConverted[:,:,2],cmap='gray',vmin=-128,vmax=127)
    plt.show()


def main():
    part1()
    part2()
    part3()


if __name__ == "__main__":
    main()
