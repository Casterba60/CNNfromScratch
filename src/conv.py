import numpy as np

class Conv3x3:
    #Convolution Layer using 3x3 filters

    def __init__ (self, num_filters):
        self.num_filters = num_filters

        #filters is a 3d array of 3x3 filters
        #dividing by 9 reduces variance -> xavier initialization
        self.filters = np.random.randn(num_filters,3,3)/9

    def iterate_regions(self,image):
        '''
        Generates all possible 3x3 image regions using valid padding.
        - image is a 2d numpy array
        '''
        #TODO: 
        #add functionality for adding padding
        #what if the image is a 3D numpy array? (color channels)

        h, w = image.shape

        for i in range(h-2):
            for j in range(w-2):
                image_region = image[i:(i+3),j:(j+3)]
                yield image_region,i,j
    
    def forward(self,input):
        '''
        Performs a forward pass of the conv layer.
        Returns a 3d numpy array with dims (h,w,num_filters)
        - input is only 2d
        '''
        #TODO:
        #need to add functionality for 3d input to stack layers

        h,w = input.shape
        output = np.zeros((h-2,w-2,self.num_filters))

        for image_region,i,j in self.iterate_regions(input):
            #creates a 1D array of the conv results, idx corresponds to filter used
            output[i,j] = np.sum(image_region*self.filters, axis=(1,2))