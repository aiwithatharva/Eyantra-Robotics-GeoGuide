'''
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 4A of Geo Guide (GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			[ GG_3303 ]
# Author List:		[ Atharva Satish Attarde, Prachit Suresh Deshinge, Ashutosh Anil Dongre, Nachiket Ganesh Apte ]
# Filename:			Task_5a_EventDetection.py


####################### IMPORT MODULES #######################
import cv2
import cv2.aruco as aruco
import sys
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

def resize_image(frame):
        """
        Resize the input frame to a predefined size while maintaining aspect ratio.

        The function takes an image frame, typically from a video feed, and resizes it to a target size 
        specified within the function. This resizing process is designed to maintain the original aspect 
        ratio of the frame to prevent distortion. The target size and the method of resizing 
        are determined based on the application's requirements.

        Args:
            frame (np.array): The input frame captured from a video source, represented as a numpy array.
        Returns:
            np.array: The resized frame as a numpy array.
        """

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
                    target_corners.append(corners[i][0][2])
                if ids[i] == 4:
                    target_corners.append(corners[i][0][3])
                if ids[i] == 6:
                    target_corners.append(corners[i][0][0])
                if ids[i] == 7:
                    target_corners.append(corners[i][0][1])
        if len(target_corners) == 4:
            target_corners = np.array(target_corners, dtype=np.int32)
            min_x, min_y = np.min(target_corners, axis=0)
            max_x, max_y = np.max(target_corners, axis=0)
            cropped_image = frame[min_y:max_y, min_x:max_x]
            height, width, _ = cropped_image.shape
            aspect_ratio = width / height
            display_width = 800
            display_height = int(display_width / aspect_ratio)
            img_resized = cv2.resize(cropped_image, (display_width, display_height))
            return img_resized
        else:
            return None
        
def deNoise(image):
    """
    Apply noise reduction techniques to an input image to improve its quality.

    This function uses filtering techniques, such as Gaussian blur, to remove noise from the image. 
    Noise reduction is an essential preprocessing step in many image processing and computer vision 
    tasks, as it can enhance the performance of subsequent algorithms by reducing the amount of 
    irrelevant or extraneous information in the image.

    Args:
        image (np.array): The input image in which noise is to be reduced. The image should be 
                          represented as a numpy array, possibly with multiple channels for color images.

    Returns:
        np.array: The denoised image as a numpy array, with the same dimensions and color depth as the 
                  input image.
    """

    sigma = 0.2
    return cv2.GaussianBlur(image, (5, 5), sigma)

def classify_event(image):
    """
    Classifies the event in the input image.

    This function utilizes a pre-trained deep learning model to analyze the input image and 
    classify it into predefined categories based on the event it depicts. The classification 
    process involves preparing the image according to the model's requirements, making a prediction, 
    and then interpreting the prediction to assign an event category to the image.

    Args:
        image (np.array): The input image to be classified, formatted as a numpy array.

    Returns:
        str: The classified event category as a string.
    """

    index_list = ["Combat", "Destroyed Buildings", "Fire", "Humanitarian Aid and rehabilitation", "Military Vehicles", "None"]
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
    model = torch.load("Model.pth", map_location=device)
    model.eval()
    with torch.no_grad():
        output = model(input_batch)
        predicted_class_index = torch.argmax(output)
    
    event = index_list[predicted_class_index]
    return event
    
    
def detectContour(image):
    """
    Detects contours in the input image and identifies relevant events or features.

    This function processes the input image to find edges and contours using advanced image processing 
    techniques. After detecting these contours, it further analyzes them to classify and identify 
    specific events or features of interest within the image. The classification can be based on the 
    shape, size, or arrangement of the contours. The function compiles these findings into a dictionary, 
    mapping identified events or features to their corresponding contours.

    Args:
        image (np.array): The input image in which contours and events are to be detected, represented as a numpy array.

    Returns:
        tuple: A tuple containing two elements:
            - A list of contours found in the image, with each contour represented as a numpy array of points.
            - A dictionary mapping identified events or features to their corresponding contours.
    """


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
                cv2.putText(image_with_contours, event, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                event_dict[string.ascii_uppercase[count]] = event
                count += 1
                
    return event_dict, image_with_contours
def task_4a_return(img_resized):
    """
    Processes the resized image for Task 4A requirements and returns relevant output.

    This function takes a resized image as input and applies a series of image processing 
    or analysis steps specific to the objectives of Task 4A. It could involve detecting features, 
    classifying elements within the image, or extracting certain information crucial for the task's 
    completion. The exact nature of the processing depends on the Task 4A specifications.

    Args:
        img_resized (np.array): The resized image to be processed, represented as a numpy array.

    Returns:
        Any: The return type and structure depend on the specific requirements of Task 4A. This 
             could be a single value, a tuple of values, or a complex data structure containing 
             the processed information or results derived from the image.
    """

    event_dict, image_with_contours = detectContour(img_resized)
    
    return event_dict, image_with_contours


###############	Main Function	#################
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
        event_dict, image_with_contours = task_4a_return(img_resized)
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
    task_4a_return(cap)
    
if __name__ == "__main__":
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    main()