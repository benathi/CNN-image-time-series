from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize
#from skimage.viewer import ImageViewer
import numpy as np

def process(im):
    orig_rows, orig_cols = im.shape
    if orig_rows < orig_cols:
        for addition in range(0,(orig_cols-orig_rows)//2):
            #adding white rows
            lst = np.array(list(float(255) for x in range(0,orig_cols)))
            im= np.vstack((im,lst))
        for addition in range(0,(orig_cols-orig_rows)//2):
            #adding white rows
            lst = np.array(list(float(255) for x in range(0,orig_cols)))
            im= np.vstack((lst,im))
    if orig_rows > orig_cols:
        for addition in range(0,(orig_rows-orig_cols)//2):
            #adding white columns
            lst = np.array(list([float(255)] for x in range(0,orig_rows)))
            im= np.hstack((im,lst))
        for addition in range(0,(orig_rows-orig_cols)//2):
            #adding white columns
            lst = np.array(list([float(255)] for x in range(0,orig_rows)))
            im= np.hstack((lst,im))
    return(im)   
    #  plt.imshow(im, cmap=cm.gray)
#    plt.show()
    
if __name__ == "__main__":
    #testNeuralNet()
    #main()   
    file = '../../data/plankton/train/amphipods/133542.jpg';
    im = imread(file, as_grey=True)
    process(im)