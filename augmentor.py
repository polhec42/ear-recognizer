import cv2
import random
import os, sys
import pandas as pd

class Data_augmentation:
    def __init__(self, path, image_name):
        '''
        Import image
        :param path: Path to the image
        :param image_name: image name
        '''
        self.path = path
        self.name = image_name
        self.image = cv2.imread(path+'/'+image_name)

    def rotate(self, image, angle=90, scale=1.0):
        '''
        Rotate the image
        :param image: image to be processed
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''
        w = image.shape[1]
        h = image.shape[0]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
        #rotate
        image = cv2.warpAffine(image,M,(w,h))
        return image

    def flip(self, image, vflip=False, hflip=False):
        '''
        Flip the image
        :param image: image to be processed
        :param vflip: whether to flip the image vertically
        :param hflip: whether to flip the image horizontally
        '''
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image 
    
    
    def image_augment(self, save_path): 
        '''
        Create the new image with imge augmentation
        :param path: the path to store the new image
        ''' 
        img = self.image.copy()
        img_flip = self.flip(img, vflip=False, hflip=True)
        img_rot = self.rotate(img, angle=15)
        img_rot1 = self.rotate(img, angle=25)
        img_rot2 = self.rotate(img, angle=35)
        #img_gaussian = self.add_GaussianNoise(img)
        
        
        cv2.imwrite(save_path+'/' + str(self.name).replace(".png", "") +'_vflip.png', img_flip)
        cv2.imwrite(save_path+'/' + str(self.name).replace(".png", "") +'_rot.png', img_rot)
        cv2.imwrite(save_path+'/' + str(self.name).replace(".png", "") +'_rot1.png', img_rot1)
        cv2.imwrite(save_path+'/' + str(self.name).replace(".png", "") +'_rot2.png', img_rot2)

        #cv2.imwrite(save_path+'%s' %str(name_int)+'_GaussianNoise.jpg', img_gaussian)
        

root_dir = 'data/ears/awe'

for subdir, dirs, files in os.walk(root_dir):
    if subdir == root_dir:
        continue
    print(subdir)
    for subdir1, dirs1, files1 in os.walk(subdir):
        for file in files1:
            if(file.lower().endswith(('.png'))):
                raw_image = Data_augmentation(subdir,file)
                raw_image.image_augment(subdir)

csv_file = 'data/ears/awe-test.csv'
readed_csv = pd.read_csv(csv_file)

output_file = open('data/ears/test.csv', 'w')

output_file.write("AWE-Full image path,AWE image path,Subject ID\n")

for i in range(0, len(readed_csv)):
    arg1 = readed_csv.iloc[i, 0]
    arg2 = readed_csv.iloc[i, 1]
    arg3 = readed_csv.iloc[i, 2]

    arg1_s = str(arg1).split(".png")
    arg2_s = str(arg2).split(".png")

    output_file.write("{},{},{}\n".format(arg1, arg2, arg3))
    output_file.write("{}_vflip.png,{}_vflip.png,{}\n".format(arg1_s[0], arg2_s[0], arg3))
    output_file.write("{}_rot.png,{}_rot.png,{}\n".format(arg1_s[0], arg2_s[0], arg3))
    output_file.write("{}_rot1.png,{}_rot1.png,{}\n".format(arg1_s[0], arg2_s[0], arg3))
    output_file.write("{}_rot2.png,{}_rot2.png,{}\n".format(arg1_s[0], arg2_s[0], arg3))

output_file.close()

"""
csv_file = 'data/ears/awe-train.csv'
readed_csv = pd.read_csv(csv_file)

output_file = open('data/ears/train.csv', 'w')

output_file.write("AWE-Full image path,AWE image path,Subject ID\n")

for i in range(0, len(readed_csv)):
    arg1 = readed_csv.iloc[i, 0]
    arg2 = readed_csv.iloc[i, 1]
    arg3 = readed_csv.iloc[i, 2]

    arg1_s = str(arg1).split(".png")
    arg2_s = str(arg2).split(".png")

    output_file.write("{},{},{}\n".format(arg1, arg2, arg3))
    output_file.write("{}_vflip.png,{}_vflip.png,{}\n".format(arg1_s[0], arg2_s[0], arg3))
    output_file.write("{}_rot.png,{}_rot.png,{}\n".format(arg1_s[0], arg2_s[0], arg3))
    output_file.write("{}_rot1.png,{}_rot1.png,{}\n".format(arg1_s[0], arg2_s[0], arg3))
    output_file.write("{}_rot2.png,{}_rot2.png,{}\n".format(arg1_s[0], arg2_s[0], arg3))

output_file.close()
"""