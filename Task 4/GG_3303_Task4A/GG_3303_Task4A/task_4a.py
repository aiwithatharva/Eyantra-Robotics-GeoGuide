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
# Filename:			task_4a.py


####################### IMPORT MODULES #######################
import cv2
import cv2.aruco as aruco
import sys
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import string
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
    Resizes the image based on ArUco markers detection.

    Args:
        frame: Input frame from the camera.

    Returns:
        Resized image or None if required markers are not found.
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
    Applies Gaussian blur to reduce noise in the image.

    Args:
        image: Input image.

    Returns:
        Denoised image.
    """
    sigma = 0.2
    return cv2.GaussianBlur(image, (5, 5), sigma)

def classify_event(image):
    """
    Classifies the event in the image using the provided model.

    Args:
        image: Image to classify.
        model: Pre-trained deep learning model.

    Returns:
        Classified event.
    """
    index_list = ["combat", "destroyed_buildings", "fire", "human_aid_rehabilitation", "military_vehicles"]
    data_transform = transforms.Compose([
    transforms.Resize((50,  50)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    pil_image = Image.fromarray(image)
    r, g, b = pil_image.split()
    r = r.point(lambda i: i * 1.0)
    g = g.point(lambda i: i * 1.1)
    b = b.point(lambda i: i * 1.1)
    result_img = Image.merge('RGB', (r, g, b))
    input_tensor = data_transform(result_img)
    input_batch = input_tensor.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("Task_4a_Model.pth", map_location=device)
    model.eval()
    with torch.no_grad():
        output = model(input_batch)
        predicted_class_index = torch.argmax(output)
    
    event = index_list[predicted_class_index]
    return event
    
    
def detectContour(image):
    """
    Detects contours in the image and classifies events at specific points.

    Args:
        image: Image for contour detection.

    Returns:
        Dictionary of events and image with contours.
    """
    arena = cv2.resize(image, (700, 700))
    image = arena
    image_with_contours = image.copy()
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=1)
    event_dict = {}
    points = [(176, 620), (515, 503), (511, 363), (171, 376), (161, 140)]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 253, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for point in points:
        px, py = point
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x <= px <= x + w and y <= py <= y + h and w < 100 and h < 100 and h > 55:
                frame = image[y:y+h, x:x+w]
                frame = frame[5:55, 5:55]
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
    Captures frames from the camera, processes them

    Args:
        cap: Video capture object.
    
    
    return: 
        event_dict: Dictionary of events detected
        image_with_contours: Image with contours
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
                print(event_dict)
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