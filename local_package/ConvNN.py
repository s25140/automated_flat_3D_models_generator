import cv2
import numpy as np
import os

def convert_labels_to_256x256(image_height, image_width, labels):
    half_height = image_height//2
    half_width = image_width//2
    new_labels = []
    for label in labels:
        old_x, old_y = label
        a_x = (half_height*11.25-half_height*8)/(54720)
        b_x = half_height/180-a_x*90
        #old_x == int(a_x*new_x**2+b_x*new_x)
        new_x = int(np.roots([a_x, b_x, -old_x])[-1])

        a_y = (half_width*11.25-half_width*8)/(54720)
        b_y = half_width/180-a_y*90
        #old_y == int(a_y*new_y**2+b_y*new_y)
        new_y = int(np.roots([a_y, b_y, -old_y])[-1])

        new_labels.append((new_x, new_y))
    
    return new_labels

    
def preproces(img_path, expodential_interpolation=True):
    source_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.equalizeHist(source_image)

    #write into a new 256x256 image pixels from the original image starting from the edges on the image
    new_image = np.zeros((256, 256), dtype=np.uint8)
    for x in range(128):
        for y in range(128):
            #write the pixels with quad growth so for the 128 pixel was the same as len(image)/2 pixels
            half_height = len(image)//2
            half_width = len(image[0])//2
            a_x = (half_height*11.25-half_height*8)/(54720)
            b_x = half_height/180-a_x*90
            old_x = int(a_x*x**2+b_x*x)
            a_y = (half_width*11.25-half_width*8)/(54720)
            b_y = half_width/180-a_y*90
            old_y = int(a_y*y**2+b_y*y)

            # calculate the new pixels color using interpolation
            new_image[x][y] = image[old_x][old_y]
            new_image[255-x][255-y] = image[len(image)-1-old_x][len(image[0])-1-old_y]
            new_image[255-x][y] = image[len(image)-1-old_x][old_y]
            new_image[x][255-y] = image[old_x][len(image[0])-1-old_y]
    image = new_image
    return image

    source_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if not expodential_interpolation:
        source_image = cv2.resize(source_image, (256, 256))

    image = cv2.equalizeHist(source_image)

    #write into a new 256x256 image pixels from the original image starting from the edges on the image
    if expodential_interpolation:
        new_image = np.zeros((256, 256), dtype=np.uint8)
        for x in range(128):
            for y in range(128):
                #write the pixels with quad growth so for the 128 pixel was the same as len(image)/2 pixels
                half_height = len(image)//2
                half_width = len(image[0])//2
                a_x = (half_height*11.25-half_height*8)/(54720)
                b_x = half_height/180-a_x*90
                old_x = int(a_x*x**2+b_x*x)
                a_y = (half_width*11.25-half_width*8)/(54720)
                b_y = half_width/180-a_y*90
                old_y = int(a_y*y**2+b_y*y)

                # calculate the new pixels color using interpolation
                new_image[x][y] = image[old_x][old_y]
                new_image[255-x][255-y] = image[len(image)-1-old_x][len(image[0])-1-old_y]
                new_image[255-x][y] = image[len(image)-1-old_x][old_y]
                new_image[x][255-y] = image[old_x][len(image[0])-1-old_y]
        image = new_image
    try:
        image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    except cv2.error:
        print('png file, no denoising')
    image = cv2.Canny(image, 8, 32, L2gradient=False)
    image2 = cv2.Canny(new_image, 50, 150, L2gradient=False)
    #combine the two edge detection results
    image = cv2.bitwise_or(image, image2)

    return image

def load_images(labels=False):
    '''load pixels from images into memory'''
    not_cropped_image_folder = "./scrapped_images1"
    cropped_image_folder = "./Cropped"

    image_files_dirs = [os.listdir(not_cropped_image_folder), os.listdir(cropped_image_folder)]

    images = []
    labels = []

    for label, image_files in enumerate(image_files_dirs):
        for file_name in image_files:
            file_path = os.path.join(cropped_image_folder if label else not_cropped_image_folder, file_name)
            try:
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                image = preproces(file_path)
            
                images.append(image)
                labels.append(label)
            except Exception as e:
                print(f"Exception for {file_name}:", e)
                
    images = np.array(images)
    labels = np.array(labels)
    # Normalize the pixel values to the range [0, 1]
    images = images / 255.0
    
    if labels:
        return images, labels
    else:
        return images