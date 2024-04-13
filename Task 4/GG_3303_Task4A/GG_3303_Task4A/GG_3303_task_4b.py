'''
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 4B of Geo Guide (GG) Theme (eYRC 2023-24).
*****************************************************************************************
'''

# Team ID:			[ GG_3303 ]
# Author List:		[ Atharva Satish Attarde, Prachit Suresh Deshinge, Ashutosh Anil Dongre, Nachiket Ganesh Apte ]
# Filename:			task_4b.py


####################### IMPORT MODULES #######################

import cv2
import cv2.aruco as aruco
import numpy as np
import pandas as pd
import csv
import socket

# WiFi credentials
ssid = "vivo T1 5G"
password = "11111111"

# ESP32 IP and port
esp_ip = "192.168.61.224"
esp_port = 8002
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Global variables
thisStop = None
direction = None
lastId = None

# Lists for coordinates and ArUco markers
crop_coordinates_list = []
aruco_marker_dict = {}

def fillArucoDict_and_cropList(frame):
    """
    Detects ArUco markers in the given frame and fills the global aruco_marker_dict and crop_coordinates_list. 

    The function searches for 52 specific markers in the frame. If found, it calculates and stores the center of each marker 
    in aruco_marker_dict and updates crop_coordinates_list with the corners of markers 5, 4, 6, and 7.

    Parameters:
    frame (numpy.ndarray): The image frame to process.

    Returns:
    bool: True if all 52 markers are found, False otherwise.
    """
    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.convertScaleAbs(gray, alpha=0.3, beta=1)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(result, aruco_dict, parameters=parameters)

    global crop_coordinates_list
    if ids is None:
        return False
    if len(ids) != 52:
        return False

    for i in range(len(ids)):
        corner = corners[i][0]
        center = np.mean(corner, axis=0)
        aruco_marker_dict[ids[i][0]] = center
        # Storing specific corners for cropping
        if ids[i] == 5:
           crop_coordinates_list .append(corner[0])
        if ids[i] == 4:
            crop_coordinates_list.append(corner[1])
        if ids[i] == 6:
            crop_coordinates_list.append(corner[2]) 
        if ids[i] == 7:
            crop_coordinates_list.append(corner[3])

    return True

 
def resizeImage_and_nearestAruro_botmarker(frame):
    """
    Resizes the image based on ArUco marker corners and finds the nearest marker to the marker with ID 0.

    The function first detects markers in the frame, then if marker 0 is found, it searches for the nearest marker within 
    a threshold distance. It also resizes the image based on crop_coordinates_list.

    Parameters:
    frame (numpy.ndarray): The image frame to process.

    Returns:
    tuple: A tuple containing the resized image, the ID of the nearest marker, and the corners of marker 0.
    """
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.convertScaleAbs(gray, alpha=0.3, beta=1)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(result, aruco_dict, parameters=parameters)

    nearest_id = None
    idZero_corners = None

    if ids is not None:
        target_index = np.where(ids == 0)[0]
        if len(target_index) > 0:
            idZero_corners = corners[target_index[0]][0]
            target_center = np.mean(idZero_corners, axis=0)
            # Compute distances to other markers
            id_distance_pairs = [(id, np.linalg.norm(target_center - aruco_marker_dict[id])) for id in range(4, 55)]
            sorted_pairs = sorted(id_distance_pairs, key=lambda pair: pair[1])
            distance_threshold = 60

            if sorted_pairs and sorted_pairs[0][1] < distance_threshold:
                nearest_id = sorted_pairs[0][0]
            elif sorted_pairs[0][0] == 7:
                nearest_id = sorted_pairs[0][0]
    # Cropping and resizing image
    target_corners = crop_coordinates_list
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
        return img_resized, nearest_id, idZero_corners

    return None, None, None



def find_and_write_nearest_csv_lat_lon(csv_data, target_id, csv_name):
    """
    Finds the latitude and longitude of the nearest marker from the CSV data and writes it to a new CSV file.

    Parameters:
    csv_data (pandas.DataFrame): DataFrame containing CSV data with latitude and longitude.
    target_id (int): The ID of the target marker.
    csv_name (str): The name of the CSV file to write the coordinates to.

    Returns:
    list or None: The latitude and longitude of the nearest marker if found, None otherwise.
    """
    
    target_id = target_id
    csv_entry = csv_data[csv_data['id'] == target_id]
    if not csv_entry.empty:
        lat = csv_entry['lat'].values[0]
        lon = csv_entry['lon'].values[0]
        coordinate = [lat, lon]
        with open(csv_name, "w", newline='') as file_name:
            csv_writer = csv.writer(file_name)
            csv_writer.writerow(["lat", "lon"])
            csv_writer.writerow(coordinate)
        return coordinate
    else:
        return None
def append_traversed_points_csv(csv_data, target_id, csv_log_name):
    """
    Appends the latitude and longitude of a traversed point to a log CSV file.

    This function adds the coordinates of the traversed point with the given ID to the log file, 
    ensuring that the same point is not logged twice consecutively.

    Parameters:
    csv_data (pandas.DataFrame): DataFrame containing CSV data with latitude and longitude.
    target_id (int): The ID of the target marker.
    csv_log_name (str): The name of the log CSV file.

    Returns:
    list or None: The appended latitude and longitude if the point is new, None otherwise.
    """
    
    csv_entry = csv_data[csv_data['id'] == target_id]
    if not csv_entry.empty:
        lat = csv_entry['lat'].values[0]
        lon = csv_entry['lon'].values[0]
        coordinate = [lat, lon]
        try:
            with open(csv_log_name, "r", newline='') as file_name:
                csv_reader = csv.reader(file_name)
                last_row = None
                for row in csv_reader:
                    last_row = row
                # Check for duplicate entries
                if last_row == [str(lat), str(lon)]: 
                    return None  
        except FileNotFoundError:
            pass  

        with open(csv_log_name, "a", newline='') as file_name:
            csv_writer = csv.writer(file_name)
            if file_name.tell() == 0:  
                csv_writer.writerow(["lat", "lon"])
            csv_writer.writerow(coordinate)
        return coordinate
    else:
        return None



def nearestStop(nearestId):
    """
    Determines the nearest stop based on the nearest marker ID.

    This function maps a given marker ID to a predefined stop in the stopDict dictionary. 
    It also accounts for special cases in navigation, like changing the stop from 'D' to 'E' when the direction is north.

    Parameters:
    nearestId (int): The ID of the nearest marker.

    Returns:
    None
    """
    
    stopDict = {
        'A' : [23, 24], 'B' : [22, 25], 'C' : [49], 'D' : [33], 'E' : [ ], 'F' : [36, 37], 'G' : [8], 'H' : [11], 'I' : [28] , 'X' : [7] # End
    }
    global thisStop
    if nearestId is not None:
        topNearestId = nearestId
        for key, values in stopDict.items():
            if topNearestId in values:
                thisStop = key
                if thisStop == 'D' and direction == 'N':
                    thisStop = 'E'
                break
    return None
def get_marker_direction(corners):
    """
    Determines the direction of a marker based on its corner points.

    The direction is calculated using the angle of the line formed by the top left and top right corners of the marker.

    Parameters:
    corners (numpy.ndarray): The corner points of the marker.

    Returns:
    None
    """
    global direction
    
    if corners is None  or len(corners) < 4:
        return
    top_left, top_right = corners[0], corners[1]
    # Calculate angle and determine direction
    angle = np.arctan2(top_right[1] - top_left[1], top_right[0] - top_left[0])
    angle_deg = np.degrees(angle)
    if -45 <= angle_deg < 45:
        direction = 'N'
    elif 45 <= angle_deg < 135:
        direction = 'E'
    elif -135 <= angle_deg < -45:
        direction = 'W'
    else:
        direction = 'S'

def send_to_esp32():
    """
    Sends the current stop and direction information to the ESP32 via UDP.
    This function constructs a message containing the current stop and direction, encodes it, and sends it to the ESP32 device 
    specified by the global variables esp_ip and esp_port. It only sends the data if both thisStop and direction are not None.

    Returns:
    None
    """
    message = f"{thisStop}{direction}"
    data = message.encode()
    if(thisStop != None and direction != None):
        sock.sendto(data, (esp_ip, esp_port))

def main():
    """
    Main function to run the marker detection and handling loop.

    This function continuously captures frames from the camera, processes them to find and handle ArUco markers, 
    and performs necessary actions based on the detected markers. It includes reading and writing CSV files, 
    resizing images, determining directions, and communicating with ESP32. It also handles displaying the processed images.

    Returns:
    None
    """
    fetched = False
    inputFile = "lat_long.csv"
    csv_data = pd.read_csv(inputFile)
    global lastId
    while True:
        ret, frame = cap.read()
        if fetched == False:
            # Initial detection of ArUco markers.
            condition = fillArucoDict_and_cropList(frame)
            if condition == True:

                fetched = True
            else:
                continue
        # Handle frame reading error.
        if not ret:
            print("Error reading frame from the camera.")
        # Process the frame to resize and find nearest marker.
        img_resized, nearestId, idZeroCorner = resizeImage_and_nearestAruro_botmarker(frame)
        if img_resized is None:
            continue
        # Determine marker direction
        get_marker_direction(idZeroCorner)
        if nearestId is not None:
            lastId = nearestId
        if lastId is not None:
            nearestStop(lastId)
            outputFile = "output.csv"
            find_and_write_nearest_csv_lat_lon(csv_data,lastId, outputFile)
            logfile = "task_4b.csv"
            append_traversed_points_csv(csv_data, lastId, logfile)
        # Send data to ESP32.
        send_to_esp32()  
        # Display the resized image.    
        cv2.namedWindow('Image with Bounding Contours', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image with Bounding Contours', 960, 960)
        cv2.moveWindow('Image with Bounding Contours', 0, 0)
        cv2.imshow('Image with Bounding Contours', img_resized)
        # Exit loop if 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release camera and close all OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # Camera setup.
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    main()
    