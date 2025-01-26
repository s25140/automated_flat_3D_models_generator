import cv2
import numpy as np
import glob
import os

debug = False


# look through pixels from the corners and find the first non-white pixel
        
def get_corners(image, MAX_RANGE_DEV=50):
    x_margin = len(image[0])/MAX_RANGE_DEV
    y_margin = len(image)/MAX_RANGE_DEV
    print(x_margin, y_margin)
    white_color_threshold = 150

    top_left, top_right, bottom_left, bottom_right = None, None, None, None
    # top left
    for y in range(int(y_margin)):
        for x in range(int(x_margin)):
            #print(x, y, image[y, x])
            if not all([i > white_color_threshold for i in image[y, x]]) and x < x_margin and y < y_margin:
                top_left = (x, y)
                break
        else:
            continue
        break
    # bottom left
    for y in range(len(image)-1, int(y_margin), -1):
        for x in range(int(x_margin)):
            #print(x, y, image[y, x])
            if not all([i > white_color_threshold for i in image[y, x]]) and x < x_margin and y > len(image)-y_margin:
                bottom_left = (x, y)
                break
        else:
            continue
        break
    # top right
    for x in range(len(image[0])-1, int(x_margin), -1):
        for y in range(int(y_margin)):
            #print(x, y, image[y, x])
            if not all([i > white_color_threshold for i in image[y, x]]) and x > len(image[0])-x_margin and y < y_margin:
                top_right = (x, y)
                break
        else:
            continue
        break
    # bottom right
    for x in range(len(image[0])-1, int(x_margin), -1):
        for y in range(len(image)-1, int(y_margin), -1):
            #print(x, y, image[y, x])
            if not all([i > white_color_threshold for i in image[y, x]]) and x > len(image[0])-x_margin and y > len(image)-y_margin:
                bottom_right = (x, y)
                break
        else:
            continue
        break

    return top_left, top_right, bottom_left, bottom_right


def crop_new_image(image, mask_image):
    # resize mask to match original image size
    mask_image = cv2.resize(mask_image, (len(image[0]), len(image)), resample=cv2.INTER_LINEAR)
    # crop
    top_left, top_right, bottom_left, bottom_right = get_corners(mask_image)
    crop_with_perspective(image, top_left, top_right, bottom_left, bottom_right)
    return image


def trim_to_edges(image, img_name):
    if not img_name.startswith('19_'):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if img_name.startswith('8_'):
            edges = cv2.Canny(gray, 10, 32)
        else:
            #print(canny_2)
            #edges = cv2.Canny(gray, canny_2[0], canny_2[1])
            edges = cv2.Canny(gray, 50, 90)#(50, 150)
            if debug:
                cv2.imshow('edges', edges)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=8, minLineLength=10, maxLineGap=10)
        #lines = cv2.HoughLinesP(edges, HoughLinesP[0], HoughLinesP[1], threshold=HoughLinesP[2], minLineLength=HoughLinesP[3], maxLineGap=HoughLinesP[4])
        canvas_edges = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            canvas_edges.append((x1, y1))
            canvas_edges.append((x2, y2))

        # Find the bounding rectangle of the canvas edges
        x, y, w, h = cv2.boundingRect(np.array(canvas_edges))

        # Crop the image to the bounding rectangle
        if img_name.startswith('5_'):
            top_left = (x+int(len(image[0])/600), y+int(len(image[0])/600))
            top_right = (x+w-int(len(image[0])/600), y+int(len(image[0])/600))
            bottom_left = (x+int(len(image[0])/600), y+h-int(len(image[0])/600))
            bottom_right = (x+w-int(len(image[0])/600), y+h-int(len(image[0])/600))
        elif img_name.startswith('6_'):
            top_left = (x-int(len(image[0])/300), y+int(len(image[0])/300))
            top_right = (x+w-int(len(image[0])/600), y+int(len(image[0])/300))
            bottom_left = (x-int(len(image[0])/300), y+h-int(len(image[0])/300))
            bottom_right = (x+w-int(len(image[0])/600), y+h-int(len(image[0])/300))
        elif img_name.startswith('7_') or img_name.startswith('12_'):
            top_left = (x+int(len(image[0])/200), y+int(len(image[0])/200))
            top_right = (x+w-int(len(image[0])/70), y+int(len(image[0])/200))
            bottom_left = (x+int(len(image[0])/200), y+h-int(len(image[0])/150))
            bottom_right = (x+w-int(len(image[0])/70), y+h-int(len(image[0])/150))
        elif img_name.startswith('8_'):
            top_left = (x, y+int(len(image[0])/600))
            top_right = (x+w, y+int(len(image[0])/600))
            bottom_left = (x, y+h)
            bottom_right = (x+w, y+h)
        elif img_name.startswith('9_') or img_name.startswith('11_'):
            top_left = (x+int(len(image[0])/600), y+int(len(image[0])/400))
            top_right = (x+w-int(len(image[0])/600), y+int(len(image[0])/400))
            bottom_left = (x+int(len(image[0])/600), y+h-int(len(image[0])/600))
            bottom_right = (x+w-int(len(image[0])/600), y+h-int(len(image[0])/600))
        elif img_name.startswith('10_') or img_name.startswith('13_'):
            top_left = (x+int(len(image[0])/600), y+int(len(image[0])/600))
            top_right = (x+w-int(len(image[0])/600), y+int(len(image[0])/600))
            bottom_left = (x+int(len(image[0])/600), y+h-int(len(image[0])/600))
            bottom_right = (x+w-int(len(image[0])/600), y+h-int(len(image[0])/600))
        elif img_name.startswith('18_'):
            top_left = (x+int(len(image[0])/300), y+int(len(image[0])/300))
            top_right = (x+w-int(len(image[0])/300), y+int(len(image[0])/300))
            bottom_left = (x+int(len(image[0])/300), y+h-int(len(image[0])/300))
            bottom_right = (x+w-int(len(image[0])/300), y+h-int(len(image[0])/300))
        else:
            top_left = (x, y)
            top_right = (x+w, y)
            bottom_left = (x, y+h)
            bottom_right = (x+w, y+h)

        if img_name.startswith('20_'):# or img_name.startswith('22_'):
            image = crop_with_perspective(image, top_left, top_right, bottom_left, bottom_right)
    
    if img_name.startswith('19_') or img_name.startswith('20_') or img_name.startswith('22_'):
        
        if img_name.startswith('20_'):
            tmp_top_left, tmp_top_right, tmp_bottom_left, tmp_bottom_right = get_corners(image)
            if None in [tmp_top_left, tmp_top_right, tmp_bottom_left, tmp_bottom_right]:
                tmp_top_left, tmp_top_right, tmp_bottom_left, tmp_bottom_right = get_corners(image, MAX_RANGE_DEV=30)
            if None in [tmp_top_left, tmp_top_right, tmp_bottom_left, tmp_bottom_right]:
                raise Exception(f'Error for {img_name}: Could not find corners with perspective stretching.')
            # find the points for the source image based on the corners
            new_top_left, new_top_right, new_bottom_left, new_bottom_right = [0, 0], [0, 0], [0, 0], [0, 0]
            new_top_left[0] = top_left[0] + tmp_top_left[0]
            new_top_left[1] = top_left[1] + tmp_top_left[1]
            new_top_right[0] = top_right[0]-len(image[0])+tmp_top_right[0]
            new_top_right[1] = top_right[1] + tmp_top_right[1]
            new_bottom_left[0] = bottom_left[0] + tmp_bottom_left[0]
            new_bottom_left[1] = bottom_left[1]-len(image)+tmp_bottom_left[1]
            new_bottom_right[0] = bottom_right[0]-len(image[0])+tmp_bottom_right[0]
            new_bottom_right[1] = bottom_right[1]-len(image)+tmp_bottom_right[1]
            top_left, top_right, bottom_left, bottom_right = new_top_left, new_top_right, new_bottom_left, new_bottom_right
        elif img_name.startswith('22_'):
            top_left, top_right, bottom_left, bottom_right = get_corners(image, MAX_RANGE_DEV=5)
            print(top_left, top_right, bottom_left, bottom_right)
            if None in [top_left, top_right, bottom_left, bottom_right]:
                top_left, top_right, bottom_left, bottom_right = get_corners(image, MAX_RANGE_DEV=5)
            if None in [top_left, top_right, bottom_left, bottom_right]:
                raise Exception(f'Error for {img_name}: Could not find corners with perspective stretching.')
        else:
            top_left, top_right, bottom_left, bottom_right = get_corners(image)
            print(top_left, top_right, bottom_left, bottom_right)
            if None in [top_left, top_right, bottom_left, bottom_right]:
                top_left, top_right, bottom_left, bottom_right = get_corners(image, MAX_RANGE_DEV=30)
            if None in [top_left, top_right, bottom_left, bottom_right]:
                raise Exception(f'Error for {img_name}: Could not find corners with perspective stretching.')

    if img_name.startswith('0_'):
        top_left = (top_left[0], 0)
        top_right = (len(image[0]), 0)
        bottom_right = (len(image[0]), bottom_right[1])

    return top_left, top_right, bottom_left, bottom_right


def crop_with_perspective(image, top_left, top_right, bottom_left, bottom_right):
    #crop image to the corners
    #image = image[top_left[1]:bottom_left[1], top_left[0]:top_right[0]]
    pts1 = np.float32([top_left, top_right, bottom_right, bottom_left])
    pts2 = np.float32([[0,0],[len(image[0]),0],[len(image[0]),len(image)],[0,len(image)]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    # view the image with the perspective lines
    if debug:
        cv2.polylines(image, [np.int32(pts1)], True, (0, 255, 0), 3)    

    image = cv2.warpPerspective(image,M,(len(image[0]),len(image)))
    return image

#def preproces():
    

# output_dir = 'D:/_projects/_for_customers/Nvzn_ML/Cropped/'

# def check_image_exists(img_name):
#     file_extension = f".{img_name.split('.')[-1]}"
#     output_path = os.path.join(output_dir, img_name.replace(file_extension, "_c" + file_extension))
#     return os.path.exists(output_path)

# def trim(batch=False):
#     if batch:
#         directory = './scrapped_images1/'
#         file_extension = '.jpg'
#         image_paths = glob.glob(directory + '*' + file_extension)
#         if not os.path.exists("./error_images.txt"):
#             open("./error_images.txt", "w").close()
        
#         for image_path in image_paths:
#             img_name = image_path.split('\\')[-1]
#             collection_index = int(img_name.split('_')[0])
#             if collection_index < 5:
#                 continue
#             image = cv2.imread(image_path)
#             #print(img_name)
#             if check_image_exists(img_name) or any(img_name.startswith(f'{str(img_type)}_') for img_type in range(14, 18)):
#                 continue
#             # Trim the image to the edges
#             try:
#                 trimmed_image = trim_to_edges(image, img_name)
#             except Exception as e:
#                 print(f'Error: {e}')
#                 with open("./error_images.txt", "a") as f:
#                     f.write(f'{img_name} -- {e}\n')
#                 continue
#             # Save the trimmed image with a new filename
#             output_path = f'{output_dir}{img_name.replace(file_extension, "_c" + file_extension)}'
#             cv2.imwrite(output_path, trimmed_image)

#             print(f'Trimmed image saved: {output_path}')
#     else: # Load the image
#         img_name = '19_163.jpg'
#         image = cv2.imread(f'./scrapped_images1/{img_name}')
#         # Trim the image to the edges
#         trimmed_image = trim_to_edges(image, img_name)
#         image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
#         trimmed_image = cv2.resize(trimmed_image, (0, 0), fx=0.5, fy=0.5)
#         # Display the original and trimmed images
#         cv2.imshow('Original Image', image)
#         cv2.imshow('Trimmed Image', trimmed_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()