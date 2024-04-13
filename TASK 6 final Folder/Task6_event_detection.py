'''
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Event Detection of Task 6 of Geo Guide (GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''


'''
* Team ID:			[ GG_3303 ]
* Author List:		[ Atharva Satish Attarde, Prachit Suresh Deshinge, Ashutosh Anil Dongre, Nachiket Ganesh Apte ]
* Filename:			Task6_event_detection.py
* Theme: Geo Guide

* Functions: resize_image(frame), deNoise(image), classify_event(image),
             detectContour(image), task_return(img_resized), main()

* Global Variables: printed, ShowCount
'''

####################### IMPORT MODULES #######################
import cv2
import cv2.aruco as aruco
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import string
import csv
from copy import deepcopy
##############################################################
printed = False
Showcount = 10
################# ADD UTILITY FUNCTIONS HERE #################

"""
You are allowed to add any number of functions to this code.
"""

##############################################################
"""
* Function Name: top_view_tranform
* Input: a (np.array), b (np.array), c (np.array), d (np.array)
* Output: matrix (np.array)
* Logic: This function takes four points as input and performs a top view transformation on them.
            It calculates the transformation matrix using the source and destination points and returns it.
* Example Call: top_view_tranform(a, b, c, d)
"""

def top_view_tranform(a, b ,c, d):

    src_pts = np.float32([c,d,a,b])
    dst_pts = np.float32([[0,0], [700, 0], [0, 700], [700, 700]])
     
    matrix = np.array([[-7.46057626e-01 ,-2.91713637e-02  ,1.13393248e+03],
                      [-1.60282140e-03,  7.86183899e-01, -1.56475440e+01],
                      [ 8.38815260e-05,  2.38837714e-05,  1.00000000e+00]])
    return matrix

"""
* Function Name: resize_image
* Input: frame (np.array)
* Output: np.array or None
* Logic: This function takes an input frame and performs the following operations:
            1. Converts the frame to grayscale.
            2. Adjusts the contrast of the grayscale image using the convertScaleAbs function.
            3. Detects ArUco markers in the adjusted grayscale image.
            4. Extracts the corners and ids of the detected markers.
            5. Selects specific target markers based on their ids.
            6. Constructs a top view transformation matrix using the target corners.
            7. Warps the original frame using the transformation matrix to obtain a resized image.
            8. Flips the resized image horizontally.
            9. Returns the resized image if all target corners are found, otherwise returns None.
* Example Call: resize_image(frame)
"""

def resize_image(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.convertScaleAbs(gray, alpha=0.3, beta=1)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(result, aruco_dict, parameters=parameters)
    target_ids = [4, 5, 6, 7]
    target_corners = []
    if ids is None:
        return None
    for i in range(len(ids)):
        if ids[i] in target_ids:
            if ids[i] == 5:
                target_corners.append(corners[i][0][0])
            if ids[i] == 4:
                target_corners.append(corners[i][0][1])
            if ids[i] == 6:
                target_corners.append(corners[i][0][2])
            if ids[i] == 7:
                target_corners.append(corners[i][0][3])
    if len(target_corners) == 4:
        a, b, c, d = target_corners
        matrix = top_view_tranform(a, b, c, d)
        img_resized = cv2.warpPerspective(frame, matrix, (700, 700))
        img_resized = cv2.flip(img_resized, 1) 
        return img_resized
    else:
        return None
     
"""
* Function Name: deNoise
* Input:
  - image (np.array): The input image in which noise is to be reduced. The image should be 
    represented as a numpy array, possibly with multiple channels for color images.
* Output:
  - np.array: The denoised image as a numpy array, with the same dimensions and color depth as the 
    input image.
* Logic:
  - Apply noise reduction techniques, specifically Gaussian blur, to the input image.
  - Noise reduction is achieved by filtering the image with a Gaussian kernel, effectively 
    smoothing out pixel variations and reducing high-frequency noise.
* Example Call: 
  denoised_image = deNoise(input_image)
"""   

def deNoise(image):
    sigma = 0.2
    return cv2.GaussianBlur(image, (5, 5), sigma)

"""
* Function Name: classify_event
* Input:
  - image (np.array): The input image to be classified, formatted as a numpy array.
* Output:
  - str: The classified event category as a string.
* Logic:
  - Utilizes a pre-trained deep learning model to classify the input image into predefined 
    categories based on the event it depicts.
  - The classification process involves preparing the image, making a prediction, and 
    interpreting the prediction to assign an event category to the image.
* Example Call: 
  event_category = classify_event(input_image)
"""

def classify_event(image):
    index_list = ["Combat", "Destroyed buildings", "Fire", "Humanitarian Aid and rehabilitation", "Military Vehicles", "None"]
    data_transform = transforms.Compose([
    transforms.Resize((50,  50)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    pil_image = Image.fromarray(image)
    r, g, b = pil_image.split()
    r = r.point(lambda i: i * 1)
    g = g.point(lambda i: i * 1.5)
    b = b.point(lambda i: i * 1)
    result_img = Image.merge('RGB', (r, g, b))
    input_tensor = data_transform(result_img)
    input_batch = input_tensor.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('Task_6_Model.pth', map_location=device)
    model.eval()
    with torch.no_grad():
        output = model(input_batch)
        predicted_class_index = torch.argmax(output)
    
    event = index_list[predicted_class_index]
    return event    
    
"""
 * Function Name: detectContour
* Input:
  - image (np.array): The input image in which contours and events are to be detected, represented as a numpy array.
* Output:
  - tuple: A tuple containing two elements:
    - A list of contours found in the image, with each contour represented as a numpy array of points.
    - A dictionary mapping identified events or features to their corresponding contours.
* Logic:
  - Processes the input image to find edges and contours using advanced image processing techniques.
  - Analyzes the detected contours to classify and identify specific events or features within the image.
  - Compiles findings into a dictionary, mapping identified events or features to their corresponding contours.
* Example Call: 

  event_dict, image_with_contours = detectContour(input_image)
"""

def detectContour(image):
    arena = cv2.resize(image, (700, 700))
    image = arena
    image_with_contours = image.copy()
    image_grey = cv2.convertScaleAbs(image, alpha=1.5, beta=1)
    event_dict = {}
    points = [(176, 620), (515, 503), (511, 363), (171, 376), (161, 140)]
    gray = cv2.cvtColor(image_grey, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    
    label_printing_dict = {
        "Combat" : "combat",
        "Destroyed buildings" :"destroyed_buildings" , "Fire":"fire",
        "Humanitarian Aid and rehabilitation" : "humanitarian_aid" ,
        "Military Vehicles" : "military_vehicles",
        "None" : "none"
    }
    for point in points:
        px, py = point
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x <= px <= x + w and y <= py <= y + h and w < 100 and h < 100 and h > 55:
                frame = image[y:y+h, x:x+w]
                height, width = frame.shape[:2]
                crop_height = crop_width = 57
                start_y = height - 5 - crop_height
                end_y = height - 5
                start_x = 5
                end_x = 5 + crop_width
                frame = frame[start_y:end_y, start_x:end_x]
                frame = deNoise(frame)
                rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                event = classify_event(rgbFrame)
                cv2.rectangle(image_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text_x = x + w // 2 - 20 
                text_y = y - 10 
                cv2.putText(image_with_contours, label_printing_dict[event], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                event_dict[string.ascii_uppercase[count]] = event
                count += 1
                
    return event_dict, image_with_contours

"""
* Function Name: task_return
* Input:
  - img_resized (np.array): The resized image to be processed, represented as a numpy array.
* Output:
  - Any: The return type and structure depend on the specific requirements of Task 4A. This could be a single value, a tuple of values, or a complex data structure containing the processed information or results derived from the image.
* Logic:
  - Calls the detectContour function to process the resized image for Task 4A requirements.
  - Returns relevant output, which may include a dictionary of identified events and an image with contours.
* Example Call:
  ```python
  event_dict, image_with_contours = task_return(resized_image)
"""

def task_return(img_resized):
    event_dict, image_with_contours = detectContour(img_resized)
    
    return event_dict, image_with_contours

###############	Main Function	#################

"""
* Function Name: main
* Input: None
* Output: None
* Logic:
  - Initializes global variables 'printed' and 'Showcount'.
  - Enters an infinite loop to continuously read frames from the camera using 'cap.read()'.
  - Resizes each frame using the 'resize_image' function.
  - Processes the resized image using the 'task_return' function to obtain event_dict and image_with_contours.
  - Displays the image with bounding contours using OpenCV if Showcount is less than 0.
  - Prints the event dictionary to the console and writes it to a CSV file named 'Event_output.csv' if it hasn't been printed before.
  - Breaks the loop if the 'q' key is pressed.
  - Releases the camera and closes all OpenCV windows.
* Example Call: main()
"""

def main():
    
    global printed
    global Showcount
    while True:
        
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from the camera.")
        img_resized = resize_image(frame)
        if img_resized is None:
            continue
        event_dict, image_with_contours = task_return(img_resized)
        if Showcount < 0:
            cv2.namedWindow('Image with Bounding Contours', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Image with Bounding Contours', 960, 960)
            cv2.moveWindow('Image with Bounding Contours', 0, 0)
            cv2.imshow('Image with Bounding Contours', image_with_contours)
            ###### Printing the Event Dictionary ######
            if printed == False:
                print_event_dict = deepcopy(event_dict)
                for key in list(print_event_dict.keys()):
                    if print_event_dict[key] == "None":
                        del print_event_dict[key]
                print(print_event_dict)
                csv_file = 'Event_output.csv'
                with open(csv_file, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=event_dict.keys())
                    writer.writeheader()
                    writer.writerow(event_dict)
                printed = True
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        Showcount = Showcount - 1
    cap.release()
    cv2.destroyAllWindows()
    task_return(cap)
    
if __name__ == "__main__":
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    main()