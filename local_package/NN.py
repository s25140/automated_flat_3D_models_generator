#Feature extraction from images -- scrapped

import os
import cv2
import numpy as np
import math as m

def extract_features(image_path, dict_=False, include_id=True):
    # 1. find corners of frame/artwork using bin search, write 2-4 color threasholds
    # 2. background of non-copped image could be 0 Alpha channel or any other solid color
    # 3. write info about colors??? 
    # Load an image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Get image dimensions
    try:
        height, width, _ = image.shape
    except ValueError:
        # convert grayscale to 3 channel
        height, width, _ = (image.shape[0], image.shape[1], 3)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    if include_id:
        data = {'id': image_path.split('/')[-1].split('.')[0]}
    else:
        data = {}
    # Measure the time to loop through the pixels
    # start_time = time.time()

    # Write boarder vars
    threashold1 = 1
    threashold2 = 10
    shadow_threashold = 100
    shadow_threashold_for_white = 7
    shadow_size = max(height, width)//75 # 5px for 300px, 10px for 600px, 20px for 1200px
    shadow_pixel_margin = 2
    max_shadow_rgb_difference = np.int16(10)
    pixel_margin1 = int(height/300)
    pixel_margin2 = int(height/400) #/900
    if pixel_margin1==0 or pixel_margin2==0: 
        pixel_margin1 = 1
        pixel_margin2 = 1

    # Top Left Corner
    cur_row_color = image[0,int(width/2)]
    for row in range(int(height/2)):
        if all(image[row][int(width/2)] == cur_row_color): continue
        if data.get('row_pixel1',None) == None\
            and (any([1 for i in range(-pixel_margin1,pixel_margin1) if any(image[row][int(width/3)] != cur_row_color) and any([1 for ind, ch in enumerate(image[row+i][int(width/3)]) if abs(np.int16(cur_row_color[ind]) - np.int16(ch)) >= threashold1])])\
            or any([1 for i in range(-pixel_margin1,pixel_margin1) if any(image[row][int(width/2)] != cur_row_color) and any([1 for ind, ch in enumerate(image[row+i][int(width/2)])if abs(np.int16(cur_row_color[ind]) - np.int16(ch)) >= threashold1])])\
            or any([1 for i in range(-pixel_margin1,pixel_margin1) if any(image[row][int(width*2/3)] != cur_row_color) and any([1 for ind, ch in enumerate(image[row+i][int(width*2/3)])if abs(np.int16(cur_row_color[ind]) - np.int16(ch)) >= threashold1])])):
                data['row_pixel1'] = row

        if any([1 for i in range(-pixel_margin2,pixel_margin2) if any(image[row][int(width/3)] != cur_row_color) and any([1 for ind, ch in enumerate(image[row+i][int(width/3)]) if abs(np.int16(cur_row_color[ind]) - np.int16(ch)) >= threashold2])])\
            or any([1 for i in range(-pixel_margin2,pixel_margin2) if any(image[row][int(width/2)] != cur_row_color) and any([1 for ind, ch in enumerate(image[row+i][int(width/2)])if abs(np.int16(cur_row_color[ind]) - np.int16(ch)) >= threashold2])])\
            or any([1 for i in range(-pixel_margin2,pixel_margin2) if any(image[row][int(width*2/3)] != cur_row_color) and any([1 for ind, ch in enumerate(image[row+i][int(width*2/3)])if abs(np.int16(cur_row_color[ind]) - np.int16(ch)) >= threashold2])]):
                data['row_pixel2'] = row
                data['shadow_top'] = row
                cur_color = image[row][int(width/2)]
                c = [np.int16(c) for c in cur_color]
                for _row in range(row+1, row+shadow_size): #todo - do for width/3, 2/3
                    if abs(c[0] - c[1]) > max_shadow_rgb_difference or abs(c[0] - c[2]) > max_shadow_rgb_difference or abs(c[1] - c[2]) > max_shadow_rgb_difference:
                        break
                    if any([1 for i in range(-shadow_pixel_margin,shadow_pixel_margin) if any(image[_row][int(width/2)] != cur_color) and any([1 for ind, ch in enumerate(image[_row+i][int(width/2)])if abs(np.int16(cur_color[ind]) - np.int16(ch)) >= shadow_threashold])]):
                        break
                    if any([1 for i in range(0,1) if any(image[_row][int(width/2)] != cur_color) and any([1 for ind, ch in enumerate(image[_row+i][int(width/2)])if np.int16(ch) - np.int16(cur_color[ind]) >= shadow_threashold_for_white])]):
                        break
                    data['shadow_top'] = _row
                    cur_color = image[_row][int(width/2)]
                    c = [np.int16(c) for c in cur_color]
                break
        
        cur_row_color = image[row][int(width/2)]

    cur_col_color = image[int(height/2),0]  
    for col in range(int(width/2)):
        if all(image[int(height/2)][col] == cur_col_color): continue
        if data.get('col_pixel1',None) == None\
            and (any([1 for i in range(-pixel_margin1,pixel_margin1) if any(image[int(height/3)][col] != cur_col_color) and any([1 for ind, ch in enumerate(image[int(height/3)][col+i]) if abs(np.int16(cur_col_color[ind]) - np.int16(ch)) >= threashold1])])\
            or any([1 for i in range(-pixel_margin1,pixel_margin1) if any(image[int(height/2)][col] != cur_col_color) and any([1 for ind, ch in enumerate(image[int(height/2)][col+i])if abs(np.int16(cur_col_color[ind]) - np.int16(ch)) >= threashold1])])\
            or any([1 for i in range(-pixel_margin1,pixel_margin1) if any(image[int(height*2/3)][col] != cur_col_color) and any([1 for ind, ch in enumerate(image[int(height*2/3)][col+i])if abs(np.int16(cur_col_color[ind]) - np.int16(ch)) >= threashold1])])):
                data['col_pixel1'] = col

        if any([1 for i in range(-pixel_margin2,pixel_margin2) if any(image[int(height/3)][col] != cur_col_color) and any([1 for ind, ch in enumerate(image[int(height/3)][col+i]) if abs(np.int16(cur_col_color[ind]) - np.int16(ch)) >= threashold2])])\
            or any([1 for i in range(-pixel_margin2,pixel_margin2) if any(image[int(height/2)][col] != cur_col_color) and any([1 for ind, ch in enumerate(image[int(height/2)][col+i])if abs(np.int16(cur_col_color[ind]) - np.int16(ch)) >= threashold2])])\
            or any([1 for i in range(-pixel_margin2,pixel_margin2) if any(image[int(height*2/3)][col] != cur_col_color) and any([1 for ind, ch in enumerate(image[int(height*2/3)][col+i])if abs(np.int16(cur_col_color[ind]) - np.int16(ch)) >= threashold2])]):
                data['col_pixel2'] = col
                data['shadow_left'] = col
                cur_color = image[int(height/2)][col]
                c = [np.int16(c) for c in cur_color]
                for _col in range(col+1, col+shadow_size): #todo - do for height/3, 2/3
                    if abs(c[0] - c[1]) > max_shadow_rgb_difference or abs(c[0] - c[2]) > max_shadow_rgb_difference or abs(c[1] - c[2]) > max_shadow_rgb_difference:
                        break
                    if any([1 for i in range(-shadow_pixel_margin,shadow_pixel_margin) if any(image[int(height/2)][_col] != cur_color) and any([1 for ind, ch in enumerate(image[int(height/2)][_col+i])if abs(np.int16(cur_color[ind]) - np.int16(ch)) >= shadow_threashold])]):
                        break
                    if any([1 for i in range(0,1) if any(image[int(height/2)][_col] != cur_color) and any([1 for ind, ch in enumerate(image[int(height/2)][_col+i])if np.int16(ch) - np.int16(cur_color[ind]) >= shadow_threashold_for_white])]):
                        break
                    data['shadow_left'] = _col
                    cur_color = image[int(height/2)][_col]
                    c = [np.int16(c) for c in cur_color]
                break
        
        cur_col_color = image[int(height/2)][col]

    # Bottom Right Corner
    cur_row_color = image[height-1,int(width/2)]
    for row in range(height-pixel_margin1-1, int(height/2), -1):
        if all(image[row][int(width/2)] == cur_row_color): continue
        if data.get('row_pixel3',None) == None\
            and (any([1 for i in range(-pixel_margin1,pixel_margin1) if any(image[row][int(width/3)] != cur_row_color) and any([1 for ind, ch in enumerate(image[row+i][int(width/3)]) if abs(np.int16(cur_row_color[ind]) - np.int16(ch)) >= threashold1])])\
            or any([1 for i in range(-pixel_margin1,pixel_margin1) if any(image[row][int(width/2)] != cur_row_color) and any([1 for ind, ch in enumerate(image[row+i][int(width/2)])if abs(np.int16(cur_row_color[ind]) - np.int16(ch)) >= threashold1])])\
            or any([1 for i in range(-pixel_margin1,pixel_margin1) if any(image[row][int(width*2/3)] != cur_row_color) and any([1 for ind, ch in enumerate(image[row+i][int(width*2/3)])if abs(np.int16(cur_row_color[ind]) - np.int16(ch)) >= threashold1])])):
                data['row_pixel3'] = row
                
        if any([1 for i in range(-pixel_margin2,pixel_margin2) if any(image[row][int(width/3)] != cur_row_color) and any([1 for ind, ch in enumerate(image[row+i][int(width/3)]) if abs(np.int16(cur_row_color[ind]) - np.int16(ch)) >= threashold2])])\
            or any([1 for i in range(-pixel_margin2,pixel_margin2) if any(image[row][int(width/2)] != cur_row_color) and any([1 for ind, ch in enumerate(image[row+i][int(width/2)])if abs(np.int16(cur_row_color[ind]) - np.int16(ch)) >= threashold2])])\
            or any([1 for i in range(-pixel_margin2,pixel_margin2) if any(image[row][int(width*2/3)] != cur_row_color) and any([1 for ind, ch in enumerate(image[row+i][int(width*2/3)])if abs(np.int16(cur_row_color[ind]) - np.int16(ch)) >= threashold2])]):
                data['row_pixel4'] = row
                data['shadow_bot'] = row
                cur_color = image[row][int(width/2)]
                c = [np.int16(c) for c in cur_color]
                for _row in range(row-1, row-shadow_size, -1): #todo - do for width/3, 2/3
                    if abs(c[0] - c[1]) > max_shadow_rgb_difference or abs(c[0] - c[2]) > max_shadow_rgb_difference or abs(c[1] - c[2]) > max_shadow_rgb_difference:
                        break
                    if any([1 for i in range(-shadow_pixel_margin,shadow_pixel_margin) if any(image[_row][int(width/2)] != cur_color) and any([1 for ind, ch in enumerate(image[_row+i][int(width/2)])if abs(np.int16(cur_color[ind]) - np.int16(ch)) >= shadow_threashold])]):
                        break
                    if any([1 for i in range(0,1) if any(image[_row][int(width/2)] != cur_color) and any([1 for ind, ch in enumerate(image[_row+i][int(width/2)])if np.int16(ch) - np.int16(cur_color[ind]) >= shadow_threashold_for_white])]):
                        break
                    data['shadow_bot'] = _row
                    cur_color = image[_row][int(width/2)]
                    c = [np.int16(c) for c in cur_color]
                break
        
        cur_row_color = image[row][int(width/2)]
        
    cur_col_color = image[int(height/2),width-1] 
    for col in range(width-pixel_margin1-1, int(width/2), -1):
        if all(image[int(height/2)][col] == cur_col_color): continue
        if data.get('col_pixel3',None) == None\
            and (any([1 for i in range(-pixel_margin1,pixel_margin1) if any(image[int(height/3)][col] != cur_col_color) and any([1 for ind, ch in enumerate(image[int(height/3)][col+i]) if abs(np.int16(cur_col_color[ind]) - np.int16(ch)) >= threashold1])])\
            or any([1 for i in range(-pixel_margin1,pixel_margin1) if any(image[int(height/2)][col] != cur_col_color) and any([1 for ind, ch in enumerate(image[int(height/2)][col+i])if abs(np.int16(cur_col_color[ind]) - np.int16(ch)) >= threashold1])])\
            or any([1 for i in range(-pixel_margin1,pixel_margin1) if any(image[int(height*2/3)][col] != cur_col_color) and any([1 for ind, ch in enumerate(image[int(height*2/3)][col+i])if abs(np.int16(cur_col_color[ind]) - np.int16(ch)) >= threashold1])])):
                data['col_pixel3'] = col

        if any([1 for i in range(-pixel_margin2,pixel_margin2) if any(image[int(height/3)][col] != cur_col_color) and any([1 for ind, ch in enumerate(image[int(height/3)][col+i]) if abs(np.int16(cur_col_color[ind]) - np.int16(ch)) >= threashold2])])\
            or any([1 for i in range(-pixel_margin2,pixel_margin2) if any(image[int(height/2)][col] != cur_col_color) and any([1 for ind, ch in enumerate(image[int(height/2)][col+i])if abs(np.int16(cur_col_color[ind]) - np.int16(ch)) >= threashold2])])\
            or any([1 for i in range(-pixel_margin2,pixel_margin2) if any(image[int(height*2/3)][col] != cur_col_color) and any([1 for ind, ch in enumerate(image[int(height*2/3)][col+i])if abs(np.int16(cur_col_color[ind]) - np.int16(ch)) >= threashold2])]):
                data['col_pixel4'] = col
                data['shadow_right'] = col
                cur_color = image[int(height/2)][col]
                c = [np.int16(c) for c in cur_color]
                for _col in range(col-1, col-shadow_size, -1): #todo - do for height/3, 2/3
                    if abs(c[0] - c[1]) > max_shadow_rgb_difference or abs(c[0] - c[2]) > max_shadow_rgb_difference or abs(c[1] - c[2]) > max_shadow_rgb_difference:
                        break
                    if any([1 for i in range(-shadow_pixel_margin,shadow_pixel_margin) if any(image[int(height/2)][_col] != cur_color) and any([1 for ind, ch in enumerate(image[int(height/2)][_col+i])if abs(np.int16(cur_color[ind]) - np.int16(ch)) >= shadow_threashold])]):
                        break
                    if any([1 for i in range(0,1) if any(image[int(height/2)][_col] != cur_color) and any([1 for ind, ch in enumerate(image[int(height/2)][_col+i])if np.int16(ch) - np.int16(cur_color[ind]) >= shadow_threashold_for_white])]):
                        break
                    data['shadow_right'] = _col
                    cur_color = image[int(height/2)][_col]
                    c = [np.int16(c) for c in cur_color]
                break
        
        cur_col_color = image[int(height/2)][col]

    #print(data)
    data['col_pixel1'] = round(data.get('col_pixel1')/width *200,7)   if data.get('col_pixel1', None)   != None else 100.00
    data['col_pixel2'] = round(data.get('col_pixel2')/width *200,7)   if data.get('col_pixel2', None)   != None else 100.00
    data['col_pixel3'] = round(data.get('col_pixel3')/width *200,7)   if data.get('col_pixel3', None)   != None else 100.00
    data['col_pixel4'] = round(data.get('col_pixel4')/width *200,7)   if data.get('col_pixel4', None)   != None else 100.00
    data['row_pixel1'] = round(data.get('row_pixel1')/height*200,7)   if data.get('row_pixel1', None)   != None else 100.00
    data['row_pixel2'] = round(data.get('row_pixel2')/height*200,7)   if data.get('row_pixel2', None)   != None else 100.00
    data['row_pixel3'] = round(data.get('row_pixel3')/height*200,7)   if data.get('row_pixel3', None)   != None else 100.00
    data['row_pixel4'] = round(data.get('row_pixel4')/height*200,7)   if data.get('row_pixel4', None)   != None else 100.00
    data['shadow_top'] = round(data.get('shadow_top')/height*200,7)   if data.get('shadow_top', None)   != None else 100.00
    data['shadow_bot'] = round(data.get('shadow_bot')/height*200,7)   if data.get('shadow_bot', None)   != None else 100.00
    data['shadow_left']= round(data.get('shadow_left')/width *200,7)  if data.get('shadow_left', None)  != None else 100.00
    data['shadow_right']=round(data.get('shadow_right')/width *200,7) if data.get('shadow_right', None) != None else 100.00
    data['top_shadow_size']=round((data['shadow_top']-data['row_pixel2'])/height*200,7)
    data['bot_shadow_size']=round((data['row_pixel4']-data['shadow_bot'])/height*200,7)
    data['left_shadow_size']=round((data['shadow_left']-data['col_pixel2'])/width*200,7)
    data['right_shadow_size']=round((data['col_pixel4']-data['shadow_right'])/width*200,7)
    ang_dist1 = m.sqrt(data.get('col_pixel1', 0)**2 + data.get('row_pixel1', 0)**2)
    data['angle_distance1'] = round(ang_dist1/height*200,3) if ang_dist1 != 0 else 0
    ang_dist2 = m.sqrt(data.get('col_pixel2', 0)**2 + data.get('row_pixel2', 0)**2)
    data['angle_distance2'] = round(ang_dist2/height*200,3) if ang_dist2 != 0 else 0
    ang_dist3 = m.sqrt(data.get('col_pixel3', 0)**2 + data.get('row_pixel3', 0)**2)
    data['angle_distance3'] = round(ang_dist3/height*200,3) if ang_dist3 != 0 else 0
    ang_dist4 = m.sqrt(data.get('col_pixel4', 0)**2 + data.get('row_pixel4', 0)**2)
    data['angle_distance4'] = round(ang_dist4/height*200,3) if ang_dist4 != 0 else 0
    data['angle_degree1'] = round(m.degrees(m.atan2(data.get('row_pixel1', 0), data.get('col_pixel1', 0))),4)
    data['angle_degree2'] = round(m.degrees(m.atan2(data.get('row_pixel2', 0), data.get('col_pixel2', 0))),4)
    data['angle_degree3'] = round(m.degrees(m.atan2(data.get('row_pixel3', 0), data.get('col_pixel3', 0))),4)
    data['angle_degree4'] = round(m.degrees(m.atan2(data.get('row_pixel4', 0), data.get('col_pixel4', 0))),4)
    
    # end_time = time.time()
    # elapsed_time = end_time - start_time

    # print(f"Time taken to loop through pixels with OpenCV: {elapsed_time} seconds") #0.01 sec
    if dict_:
        return data
    else:
        return list(data.values())


def get_features_count():
    '''get features count'''
    # create temp image to get features count
    temp_image = np.zeros((200,200,3), np.uint8)
    # save temp image
    cv2.imwrite('temp.jpg', temp_image)
    # get features count
    features_count = len(extract_features('temp.jpg', include_id=False))
    # remove temp image
    os.remove('temp.jpg')
    return features_count
