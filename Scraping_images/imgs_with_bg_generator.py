import cv2
import numpy as np
import math
import os

def add_bg(img, bg, orig_image_coordinates=False):
    # randomize the brightness of the background
    bg = cv2.convertScaleAbs(bg, alpha=0.5, beta=np.random.randint(0, 100))
    # move the background to random direction with tiling
    bg = np.roll(bg, np.random.randint(-100, 100), axis=0)
    # add white BW noise to the background
    if np.random.rand() > 0.23:
        # create a random noise image with equal rgb values for each pixel
        # random int from 2 to 8
        noise_res_div = np.random.randint(2, 5)
        noise = np.random.randint(0,256, (bg.shape[0]//noise_res_div, bg.shape[1]//noise_res_div, 1), dtype=np.uint8)
        noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
        noise = cv2.resize(noise, (bg.shape[1], bg.shape[0]))
        # generate random weight for the noise from 0.4 to 1.0
        weight = np.random.rand() * 0.6 + 0.4
        bg = cv2.addWeighted(bg, weight, noise, 1-weight, 0)
    margin_size = img.shape[0] // 70
    shadow_size = int(margin_size / 2)
    is_float = margin_size/2 - int(margin_size/2) > 0
    # add shadow on the edges of the source image
    shadow = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # randomly align the shadow. top, left, bottom, right, top-left, top-right, bottom-left, bottom-right
    align = np.random.choice(['top', 'left', 'bottom', 'right', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'none'])
    if align == 'top':
        shadow = cv2.copyMakeBorder(shadow, shadow_size, 0, 0, 0, cv2.BORDER_CONSTANT, value=[128, 128, 128])
        shadow = cv2.copyMakeBorder(shadow, shadow_size+1 if is_float else shadow_size, 0, 0, 0, cv2.BORDER_CONSTANT, value=[256, 256, 256])
        img = cv2.copyMakeBorder(img, margin_size, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif align == 'left':
        shadow = cv2.copyMakeBorder(shadow, 0, 0, shadow_size, 0, cv2.BORDER_CONSTANT, value=[128, 128, 128])
        shadow = cv2.copyMakeBorder(shadow, 0, 0, shadow_size+1 if is_float else shadow_size, 0, cv2.BORDER_CONSTANT, value=[256, 256, 256])
        img = cv2.copyMakeBorder(img, 0, 0, margin_size, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif align == 'bottom':
        shadow = cv2.copyMakeBorder(shadow, 0, shadow_size, 0, 0, cv2.BORDER_CONSTANT, value=[128, 128, 128])
        shadow = cv2.copyMakeBorder(shadow, 0, shadow_size+1 if is_float else shadow_size, 0, 0, cv2.BORDER_CONSTANT, value=[256, 256, 256])
        img = cv2.copyMakeBorder(img, 0, margin_size, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif align == 'right':
        shadow = cv2.copyMakeBorder(shadow, 0, 0, 0, shadow_size, cv2.BORDER_CONSTANT, value=[128, 128, 128])
        shadow = cv2.copyMakeBorder(shadow, 0, 0, 0, shadow_size+1 if is_float else shadow_size, cv2.BORDER_CONSTANT, value=[256, 256, 256])
        img = cv2.copyMakeBorder(img, 0, 0, 0, margin_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif align == 'top-left':
        shadow = cv2.copyMakeBorder(shadow, shadow_size, 0, shadow_size, 0, cv2.BORDER_CONSTANT, value=[128, 128, 128])
        shadow = cv2.copyMakeBorder(shadow, shadow_size+1 if is_float else shadow_size, 0, shadow_size+1 if is_float else shadow_size, 0, cv2.BORDER_CONSTANT, value=[256, 256, 256])
        img = cv2.copyMakeBorder(img, margin_size, 0, margin_size, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif align == 'top-right':
        shadow = cv2.copyMakeBorder(shadow, shadow_size, 0, 0, shadow_size, cv2.BORDER_CONSTANT, value=[128, 128, 128])
        shadow = cv2.copyMakeBorder(shadow, shadow_size+1 if is_float else shadow_size, 0, 0, shadow_size+1 if is_float else shadow_size, cv2.BORDER_CONSTANT, value=[256, 256, 256])
        img = cv2.copyMakeBorder(img, margin_size, 0, 0, margin_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif align == 'bottom-left':
        shadow = cv2.copyMakeBorder(shadow, 0, shadow_size, shadow_size, 0, cv2.BORDER_CONSTANT, value=[128, 128, 128])
        shadow = cv2.copyMakeBorder(shadow, 0, shadow_size+1 if is_float else shadow_size, shadow_size+1 if is_float else shadow_size, 0, cv2.BORDER_CONSTANT, value=[256, 256, 256])
        img = cv2.copyMakeBorder(img, 0, margin_size, margin_size, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif align == 'bottom-right':
        shadow = cv2.copyMakeBorder(shadow, 0, shadow_size, 0, shadow_size, cv2.BORDER_CONSTANT, value=[128, 128, 128])
        shadow = cv2.copyMakeBorder(shadow, 0, shadow_size+1 if is_float else shadow_size, 0, shadow_size+1 if is_float else shadow_size, cv2.BORDER_CONSTANT, value=[256, 256, 256])
        img = cv2.copyMakeBorder(img, 0, margin_size, 0, margin_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        pass

    shadow = cv2.GaussianBlur(shadow, (shadow_size*2+1, shadow_size*2+1), shadow_size//4+1)
    
    # cv2.imshow('shadow', shadow)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # add inverted shadow to alpha channel of the source image
    if img.shape[2] < 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    

    img[:, :, 3] = 255 - shadow[:, :, 0]
    img_width = img.shape[1]
    img_height = img.shape[0]

    # resize the main image to have a 25% margin between the image and the background width
    bg_width = bg.shape[1]
    bg_height = bg.shape[0]
    new_img_width = int(bg_width * 0.8)
    new_img_height = int(img_height * new_img_width / img_width)
    img = cv2.resize(img, (new_img_width, new_img_height))
    
    # add the image on top of the background with alpha blending
    img_with_bg = bg.copy()
    x_offset = int((bg_width - new_img_width) / 2)
    y_offset = int((bg_height - new_img_height) / 2)
    while x_offset <= 0 or y_offset <= 0:
        img = cv2.resize(img, (int(new_img_width * 0.9), int(new_img_height * 0.9)))
        new_img_width = img.shape[1]
        new_img_height = img.shape[0]
        x_offset = int((bg_width - new_img_width) / 2)
        y_offset = int((bg_height - new_img_height) / 2)

    print(img.shape, img_with_bg.shape, x_offset, y_offset, new_img_width, new_img_height, align)

    img_with_bg[y_offset:y_offset + new_img_height, x_offset:x_offset + new_img_width] = img[:, :, :3] * (img[:, :, 3:] / 255) + img_with_bg[y_offset:y_offset + new_img_height, x_offset:x_offset + new_img_width] * (1 - img[:, :, 3:] / 255)

    # resize the image to 700px on the smallest side
    # if img_with_bg.shape[0] < img_with_bg.shape[1]:
    #     img_with_bg = cv2.resize(img_with_bg, (700, int(700 * img_with_bg.shape[0] / img_with_bg.shape[1])))
    # else:
    #     img_with_bg = cv2.resize(img_with_bg, (int(700 * img_with_bg.shape[1] / img_with_bg.shape[0]), 700))

    if orig_image_coordinates:
        # get the coordinates of the image on the background without the shadow margin
        x1 = x_offset
        y1 = y_offset
        x2 = x1 + new_img_width
        y2 = y1
        x3 = x2
        y3 = y1 + new_img_height
        x4 = x1
        y4 = y3
        margin_size = img_with_bg.shape[0]//70
        # remove shadow margin
        if 'top' in align:
            y1 += margin_size
            y2 += margin_size
        if 'left' in align:
            x1 += margin_size
            x4 += margin_size
        if 'bottom' in align:
            y3 -= margin_size
            y4 -= margin_size
        if 'right' in align:
            x2 -= margin_size
            x3 -= margin_size
        # add the coordinates to the image
        # img_with_bg = cv2.circle(img_with_bg, (x1, y1), 1, (0, 0, 255), -1)
        # img_with_bg = cv2.circle(img_with_bg, (x2, y2), 5, (0, 0, 255), -1)
        # img_with_bg = cv2.circle(img_with_bg, (x3, y3), 10, (0, 0, 255), -1)
        # img_with_bg = cv2.circle(img_with_bg, (x4, y4), 20, (0, 0, 255), -1)
        #top_left, top_right, bottom_right, bottom_left
        return img_with_bg, x1, y1, x2, y2, x3, y3, x4, y4

    return img_with_bg


# pick random images from each image type and add randomly selected background
def generate_images():
    image_paths = []
    img_type_iterators = {i:0 for i in range(0, 25)}
    img_type_iterators[27]=131 # for preserving the dataset balance
    for img in os.listdir('./Cropped'):
        if img_type_iterators[int(img.split('_')[0])] >= 130:
            continue
        image_paths.append('./Cropped/' + img)
        img_type_iterators[int(img.split('_')[0])] += 1

    bg_paths = []
    for bg in os.listdir('./Scraping_images/background_textures'):
        bg_paths.append('./Scraping_images/background_textures/' + bg)

    img_ind = 0
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        bg = cv2.imread(bg_paths[np.random.randint(0, len(bg_paths))])
        img_with_bg = add_bg(image, bg)
        cv2.imwrite(f'./scrapped_images1/25_{img_ind}.jpg', img_with_bg)
        img_ind-=-1
    
def generate_images_and_coordinates():
    image_paths = []
    img_type_iterators = {i:0 for i in range(0, 25)}
    img_type_iterators[27]=131 # for preserving the dataset balance
    for img in os.listdir('./Cropped'):
        if img_type_iterators[int(img.split('_')[0])] >= 130:
            continue
        image_paths.append('./Cropped/' + img)
        img_type_iterators[int(img.split('_')[0])] += 1

    bg_paths = []
    for bg in os.listdir('./Scraping_images/background_textures'):
        bg_paths.append('./Scraping_images/background_textures/' + bg)

    img_ind = 0
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        bg = cv2.imread(bg_paths[np.random.randint(0, len(bg_paths))])
        img_with_bg, x1, y1, x2, y2, x3, y3, x4, y4 = add_bg(image, bg, True)
        with open(f'./Scraping_images/generated_with_coordinates(fixed_0)/coordinates.txt', 'a') as f:
            f.write(f'25_{img_ind}.jpg {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n')
        cv2.imwrite(f'./Scraping_images/generated_with_coordinates(fixed_0)/25_{img_ind}.jpg', img_with_bg)
        img_ind-=-1
    
def debug():
    img = cv2.imread('./Datasets/cropped_images/7_1_c.jpg', cv2.IMREAD_UNCHANGED)
    # pick random background
    bg = os.listdir('./Scraping_images/background_textures')
    bg = cv2.imread('./Scraping_images/background_textures/' + bg[np.random.randint(0, len(bg))])
    #bg = cv2.imread('./Scraping_images/background_textures/Brick_Non_Uniform_12inch_Soldier_bump.png')
    img_with_bg = add_bg(img, bg, True)[0]
    cv2.imshow('Original image', img)
    cv2.imshow('Augmented image', img_with_bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #generate_images_and_coordinates()
    #generate_images()
    debug()