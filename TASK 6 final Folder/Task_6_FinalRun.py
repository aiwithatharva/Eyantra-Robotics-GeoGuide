'''
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 6 of Geo Guide (GG) Theme (eYRC 2023-24).
*****************************************************************************************
'''
'''
* Team ID:			[ GG_3303 ]
* Author List:		[ Atharva Satish Attarde, Prachit Suresh Deshinge, Ashutosh Anil Dongre, Nachiket Ganesh Apte ]
* Filename:			Task_6_FinalRun.py
* Theme: Geo Guide
* Functions: aruco_dict_ret(frame), read_event_dict(), get_priority(event), get_position(event), 
            floyd_warshall(Adj_matrix), reconstruct_path(start, end, next_vertex),
            reconstruct_combined_path(event_priority, your_dictionary, next_vertex),
            fillArucoDict_and_cropList(frame, corners, ids), arena_to_qgis_mapping(transformed_image, matrix), 
            resizeImage_and_nearestAruro_botmarker(frame, corners, ids), get_marker_direction(corners),
            get_marker_center(corners, marker_id, ids), detect_and_estimate_turning_direction(img_resized, corners, ids),
            write_lat_lon_to_output_csv(lat_lon_data, csv_name), append_traversed_points_csv(csv_data, target_id, csv_log_name),
            nearestStop(nearestId), send_to_esp32(), update_path_list_pointer_and_next_stop(), top_view_tranform(), qgis_tranform(tranformed_image), main()

* Global Variables: ssid, password, esp_ip, esp_port, sock, thisStop, nextStop, turnDirection, lastId,
                    nextStopAngle, top_view_transform_matrix, current_pointer, path_list, mask,
                    crop_coordinates_list, aruco_marker_dict, stops_only_dict, inf, Adj_matrix,
                    id_mapping_dictionary, event_priority
'''

####################### IMPORT MODULES #######################

import cv2
import cv2.aruco as aruco
import numpy as np
import pandas as pd
import csv
import time
import sys
import socket

# WiFi credentials
ssid = "abc"
password = "11111111"

# ESP32 IP and port
esp_ip = "192.168.133.224"
esp_port = 4210
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Global variables
thisStop = None
nextStop = None
turnDirection = None
lastId = None
nextStopAngle = None

firstPakageSend = False


top_view_tranform_matrix = np.array([[ 9.18090670e-01,  5.22963040e-02, -4.94978706e+02],
                                     [-2.74263685e-03,  8.96842251e-01, -1.64613064e+01],
                                     [ 8.25931965e-05,  2.37273637e-05,  1.00000000e+00]])
# Path list and pointer
current_pointer = 0
path_list = []
mask = []


# Lists for coordinates and ArUco markers
crop_coordinates_list = []
aruco_marker_dict = {}
stops_only_dict = {}

inf = float('inf')
Adj_matrix = np.load('adjacency_matrix.npy')
id_mapping_dictionary = {    
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'a',
    6: 'b',
    7: 'c',
    8: 'd',
    9: 'e',
    10: 'f',
    11: 'g',
    12: 'h',
    13: 'i',
    14: 'j',
    15: 'k',
    16: 'x'
}


event_priority = {
    'Fire': {'priority': 1, 'position': None},
    'Destroyed buildings': {'priority': 2, 'position': None}, 
    'Humanitarian Aid and rehabilitation': {'priority': 3, 'position': None},  
    'Military Vehicles': {'priority': 4, 'position': None},
    'Combat': {'priority': 5, 'position': None}  
}

"""
* Function Name: aruco_dict_ret
* Input: frame -> frame captured from the camera
* Output: Returns ArUco dictionary for the given frame
* Logic: Utilizes OpenCV's ArUco marker detection library to identify markers in the frame and return the corresponding dictionary
* Example call: aruco_dict_ret(frame)
"""

def aruco_dict_ret(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.convertScaleAbs(gray, alpha=0.3, beta=1)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(result, aruco_dict, parameters=parameters)
    return (corners, ids)

"""
* Function Name: read_event_dict
* Input: None
* Output: Returns an event dictionary
* Logic: Reads and retrieves an event dictionary, from an external source.
* Example call: read_event_dict()
"""

def read_event_dict():

    global event_priority
    with open('Event_output.csv') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        event_dict = next(csv_reader, None)
        if event_dict:
            for location in event_dict:
                event = event_dict[location]
                if event != 'None':
                    event_priority[event] = {'priority': event_priority[event]['priority'], 'position': location}

"""
* Function Name: get_priority
* Input: event (string)
* Output: Priority value corresponding to the given event.
* Logic:
  - Takes an event as input and retrieves its priority value from the `event_priority` dictionary.
* Example Call: get_priority("some_event")
"""

def get_priority(event):
    return event_priority[event]['priority']

"""
* Function Name: get_position
* Input: None
* Output: Returns the current position of the robot
* Logic: Retrieves and provides information about the current position of the robot, possibly using sensors or tracking systems
* Example call: get_position()
"""

def get_position(event):
    return event_priority[event]['position']

"""
* Function Name: flyod_warshall
* Input: distance_matrix -> matrix representing distances between different points or nodes
* Output: Returns the matrix with shortest paths between all pairs of nodes
* Logic: Applies the Floyd-Warshall algorithm to find the shortest paths between all pairs of nodes in the given distance matrix
* Example call: flyod_warshall(distance_matrix)
"""

def floyd_warshall(Adj_matrix):
    n = len(Adj_matrix)
    dist = [[float('inf') for _ in range(n)] for _ in range(n)]
    next_vertex = [[None for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if Adj_matrix[i][j] != 0:
                dist[i][j] = Adj_matrix[i][j]
                next_vertex[i][j] = j
    for i in range(n):
        dist[i][i] = 0
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_vertex[i][j] = next_vertex[i][k]
    
    return dist, next_vertex

"""
* Function Name: reconstruct_path
* Input: parent_matrix -> matrix representing the parent nodes for each node in the shortest paths
         start -> starting node
         end -> ending node
* Output: Returns the shortest path from the starting node to the ending node based on the parent matrix
* Logic: Reconstructs the shortest path from the starting node to the ending node using the parent matrix generated by the Floyd-Warshall algorithm
* Example call: reconstruct_path(parent_matrix, start, end)
"""

def reconstruct_path(start, end, next_vertex):
    mask = []
    if next_vertex[start][end] is None:
        return None
    path = [start]
    while start != end:
        start = next_vertex[start][end]
        mask.append(0)
        path.append(start)
    mask.append(1)   # Set the last vertex to 1
    return path, mask

"""
* Function Name: reconstruct_combined_path
* Input: shortest_path1 -> list representing the first shortest path
         shortest_path2 -> list representing the second shortest path
* Output: Returns a combined path that represents the union of multiple  paths
* Logic: Combines the two input paths while avoiding duplicate nodes to create a single path
* Example call: reconstruct_combined_path(shortest_path1, shortest_path2)
"""

def reconstruct_combined_path(event_priority, your_dictionary, next_vertex):
   
    global mask
    start_position = 'x'
    i=0
    event_prev = None
    combined_path = [] 
    mask = []
    i = 0
    
    # Find the first event with a non-None position
    for event, info in event_priority.items():
        if info['position'] is not None:
            if i!=0:
                start_position = get_position(event_prev)
            end_position= get_position(event)
            start = next(key for key, value in your_dictionary.items() if value == start_position)
            end = next(key for key, value in your_dictionary.items() if value == end_position)
            path1, mask1 = reconstruct_path(start, end, next_vertex)
            event_prev=event    
            if i!=0:
                path1.pop(0)
                mask1.pop(0)
            combined_path = combined_path + path1
            mask = mask + mask1
            i += 1
    start = 16
    path3, mask3 = reconstruct_path(end, start, next_vertex)
    path3.pop(0)
    mask3.pop(0)
    mask3.append(0)
    path3.append(0)
    combined_path += path3
    mask += mask3
    return combined_path

"""
* Function Name: fillArucoDict_and_cropList
* Input:
  - frame: The input frame.
  - corners: The corners of the detected markers.
  - ids: The ids of the detected markers.
* Output: 
  - True if the aruco_marker_dict and crop_coordinates_list are successfully filled, False otherwise.
* Logic:
  - This function processes information from detected ArUco markers in a given frame.
  - It populates the global aruco_marker_dict and crop_coordinates_list with marker IDs and corresponding corner coordinates, respectively.
  - Checks if valid marker IDs are present; if ids is None or has fewer than 52 elements, the function returns False as the minimum required markers are not detected.
  - Iterates through detected markers, calculates the center from their corners, and updates the aruco_marker_dict with the center coordinates if the marker ID is not 100 (excluding a specific marker ID from being processed).
  - Stores specific corners of markers with certain IDs into crop_coordinates_list, which is presumably used for further processing or analysis.
  - Calculates weighted averages of specific marker centers and stores them in the stops_only_dict dictionary with corresponding keys ('a' to 'E' and 'x'). The weights and markers used in each calculation are specific to the application's logic.
* Example Call: fillArucoDict_and_cropList(frame, corners, ids)
"""

def fillArucoDict_and_cropList(frame, corners, ids):
    global crop_coordinates_list
    if ids is None:
        return False
    if len(ids) < 52:
        return False

    for i in range(len(ids)):
        corner = corners[i][0]
        center = np.mean(corner, axis=0)
        if ids[i]!=100:
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
    weights = np.array([0.5, 0.5, 0.27], dtype=np.float32)
    stops_only_dict['a'] = np.array(np.average([aruco_marker_dict[23], aruco_marker_dict[24],aruco_marker_dict[21]], axis=0, weights=weights), dtype=np.float32)

    weights = np.array([0.6, 0.6, 0.7], dtype=np.float32)
    stops_only_dict['b'] = np.array(np.average([aruco_marker_dict[22], aruco_marker_dict[49],aruco_marker_dict[25]], axis=0, weights=weights), dtype=np.float32)

    weights = np.array([0.6, 2.9, 0.8], dtype=np.float32)
    stops_only_dict['c'] = np.array(np.average([aruco_marker_dict[49], aruco_marker_dict[50],aruco_marker_dict[34]], axis=0, weights=weights), dtype=np.float32)

    weights = np.array([0.6, 0.5, 0.9], dtype=np.float32)
    stops_only_dict['d'] = np.array(np.average([aruco_marker_dict[51], aruco_marker_dict[52],aruco_marker_dict[42]], axis=0, weights=weights), dtype=np.float32)

    weights = np.array([0.5, 0.65, 0.05], dtype=np.float32)
    stops_only_dict['e'] = np.array(np.average([aruco_marker_dict[36], aruco_marker_dict[10],aruco_marker_dict[8]], axis=0, weights=weights), dtype=np.float32)

    weights = np.array([0.52, 0.6, 0.22], dtype=np.float32)
    stops_only_dict['f'] = np.array(np.average([aruco_marker_dict[30], aruco_marker_dict[8],aruco_marker_dict[9]], axis=0, weights=weights), dtype=np.float32)

    weights = np.array([0.5, 0.525, 0.05], dtype=np.float32)
    stops_only_dict['g'] = np.array(np.average([aruco_marker_dict[29], aruco_marker_dict[11],aruco_marker_dict[13]], axis=0, weights=weights), dtype=np.float32)

    weights = np.array([1.2, 0.37, 0.08], dtype=np.float32)
    stops_only_dict['h'] = np.array(np.average([aruco_marker_dict[19], aruco_marker_dict[27],aruco_marker_dict[28]], axis=0, weights=weights), dtype=np.float32)

    weights = np.array([0.65, 0.35, 0.25], dtype=np.float32)
    stops_only_dict['i'] = np.array(np.average([aruco_marker_dict[27], aruco_marker_dict[32],aruco_marker_dict[28]], axis=0, weights=weights), dtype=np.float32)

    weights = np.array([0.65, 0.48, 0.22], dtype=np.float32)
    stops_only_dict['j'] = np.array(np.average([aruco_marker_dict[33], aruco_marker_dict[35],aruco_marker_dict[32]], axis=0, weights=weights), dtype=np.float32)

    weights = np.array([0.58, 0.58, 0.5], dtype=np.float32)
    stops_only_dict['k'] = np.array(np.average([aruco_marker_dict[39], aruco_marker_dict[35],aruco_marker_dict[46]], axis=0, weights=weights), dtype=np.float32)

    weights = np.array([0.58, 1.1, 0.5], dtype=np.float32)
    stops_only_dict['A'] = np.array(np.average([aruco_marker_dict[23], aruco_marker_dict[21],aruco_marker_dict[26]], axis=0, weights=weights), dtype=np.float32)

    weights = np.array([0.9, 0.5, 0.5], dtype=np.float32)
    stops_only_dict['B'] = np.array(np.average([aruco_marker_dict[28], aruco_marker_dict[29],aruco_marker_dict[30]], axis=0, weights=weights), dtype=np.float32)

    weights = np.array([0.6, 2.0, 1.3], dtype=np.float32)
    stops_only_dict['C'] = np.array(np.average([aruco_marker_dict[30], aruco_marker_dict[31],aruco_marker_dict[36]], axis=0, weights=weights), dtype=np.float32)

    weights = np.array([1.0, 0.5, 0.4], dtype=np.float32)
    stops_only_dict['D'] = np.array(np.average([aruco_marker_dict[34], aruco_marker_dict[50],aruco_marker_dict[41]], axis=0, weights=weights), dtype=np.float32)

    weights = np.array([0.1, 1.3, 1.0], dtype=np.float32)
    stops_only_dict['E'] = np.array(np.average([aruco_marker_dict[51], aruco_marker_dict[54],aruco_marker_dict[48]], axis=0, weights=weights), dtype=np.float32)

    weights = np.array([0.55, 0.2, -0.15], dtype=np.float32)
    stops_only_dict['x'] = np.array(np.average([aruco_marker_dict[23], aruco_marker_dict[21],aruco_marker_dict[26]], axis=0, weights=weights), dtype=np.float32)
    
    weights = np.array([0.7, 0.5, 0.5], dtype=np.float32)
    aruco_marker_dict[55] = np.array(np.average([aruco_marker_dict[27], aruco_marker_dict[19],aruco_marker_dict[28]], axis=0, weights=weights), dtype=np.float32)

    weights = np.array([0.9, 0.5, 0.38], dtype=np.float32)
    aruco_marker_dict[56] = np.array(np.average([aruco_marker_dict[33], aruco_marker_dict[28],aruco_marker_dict[32]], axis=0, weights=weights), dtype=np.float32)
    
    weights = np.array([0.9, 0.5, 0.55], dtype=np.float32)
    aruco_marker_dict[57] = np.array(np.average([aruco_marker_dict[39], aruco_marker_dict[32],aruco_marker_dict[35]], axis=0, weights=weights), dtype=np.float32)
    return True

"""
* Function Name: arena_to_qgis_mapping
* Input:
  - transformed_image (numpy.ndarray): The transformed image, obtained through image processing techniques.
  - matrix (numpy.ndarray): The transformation matrix used to map points from the transformed image to the original image.
* Output:
  - numpy.ndarray: The corresponding point in the original image for a given point in the transformed image.
* Logic:
  - This function takes a transformed image of the arena and the corresponding transformation matrix as input.
  - Writes the transformed image to a file named 'Transformed.jpg'.
  - Detects ArUco markers (corners and IDs) in the transformed image using the aruco_dict_ret function.
  - Identifies the specific marker (ID=100) used as a reference point for mapping.
  - Calculates the center of the reference marker in the transformed image.
  - Constructs a point in the transformed image using the center of the reference marker.
  - Uses the inverse of the transformation matrix to find the corresponding point in the original image.
  - Returns the corresponding point in the original image.
* Example Call: arena_to_qgis_mapping(transformed_image, matrix)
"""

def arena_to_qgis_mapping(transformed_image, matrix):
    corners, ids = aruco_dict_ret(transformed_image)
    if ids is None:
        return None
    index = (ids == 100).nonzero()[0]
    if index.size == 0:
        return None
    center = np.mean(corners[index[0]][0], axis=0)
    # Now, for any point in the transformed image, you can find its corresponding point in the original image
    point_in_transformed_image = np.array([np.array([center], dtype='float32')])
    # Use the inverse of the transformation matrix to find the corresponding point in the original image
    point_in_original_image = cv2.perspectiveTransform(point_in_transformed_image, np.linalg.inv(matrix))
    return point_in_original_image[0]
    
"""
* Function Name: resizeImage_and_nearestAruro_botmarker
* Input:
  - frame (numpy.ndarray): The input image frame captured from a camera.
  - corners (numpy.ndarray): The corners of the detected Aruco markers.
  - ids (numpy.ndarray): The IDs of the detected Aruco markers.
* Output:
  - tuple: A tuple containing the resized image, the nearest ID marker, the corners of the ID marker, and the nearest Aruco marker.
* Logic:
  - Checks if Aruco markers (corners and IDs) are detected in the input image.
  - Identifies a specific marker (ID=100) as a reference point for resizing and finding the nearest markers.
  - Computes the center of the reference marker and calculates distances to other markers, both ID markers and Aruco markers.
  - Selects the nearest ID marker based on distance, avoiding certain IDs and a distance threshold.
  - Resizes and crops the image based on predefined coordinates (crop_coordinates_list).
  - Returns a tuple containing the resized image, the nearest ID marker, the corners of the ID marker, and the nearest Aruco marker.
* Example Call: resizeImage_and_nearestAruro_botmarker(frame, corners, ids)
"""   

def resizeImage_and_nearestAruro_botmarker(frame, corners, ids):
    nearest_id = None
    idZero_corners = None
    nearest_Aruco_id = None
    if ids is not None:

        target_index = np.where(ids == '100')[0]
        if len(target_index) > 0:
            idZero_corners = corners[target_index[0]][0]
            target_center = np.mean(idZero_corners, axis=0)
            # Compute distances to other markers
            id_distance_pairs = [(id, np.linalg.norm(target_center - stops_only_dict[id])) for id in stops_only_dict if id != '100']     
            sorted_pairs = sorted(id_distance_pairs, key=lambda pair: pair[1])
            
            distance_threshold = 10000
            
            if sorted_pairs and sorted_pairs[0][1] < distance_threshold:
                nearest_id = sorted_pairs[0][0]
            if nearest_id in ['A', 'B', 'C', 'D', 'E'] and sorted_pairs[0][1] > 40:
                nearest_id = None
                
            Aruco_id_distance_pairs = [(id, np.linalg.norm(target_center - aruco_marker_dict[id])) for id in aruco_marker_dict]
            newSortedpair = sorted(Aruco_id_distance_pairs, key=lambda pair: pair[1])
            if newSortedpair and newSortedpair[0][1] < 80:
                nearest_Aruco_id = newSortedpair[0][0]
            
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
        return img_resized, nearest_id, idZero_corners, nearest_Aruco_id

    return None, None, None, None

"""
* Function Name: get_marker_direction
* Input:
  - corners (list): List of corner points of the marker.
* Returns:
  - str: The direction of the marker ('N', 'E', 'W', 'S').
* Logic:
  - Calculates the direction of a marker based on its corner points.
  - Extracts the top-left and top-right corner points from the provided list.
  - Calculates the angle of the marker using arctan2.
  - Determines the direction ('N', 'E', 'W', 'S') based on the angle.
* Example Call: get_marker_direction(corners)
"""

def get_marker_direction(corners):
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
    return direction

"""
* Function Name: get_marker_center
* Input:
  - corners (numpy.ndarray): Array of marker corners.
  - marker_id (int): ID of the marker.
  - ids (numpy.ndarray): Array of marker IDs.
* Returns:
  - tuple or None: The center coordinates of the marker as a tuple (x, y), or None if the marker is not found.
* Logic:
  - Finds the index of the specified marker_id in the array of marker IDs.
  - If the marker_id is found, extracts the corner points of the marker and calculates its center.
  - Returns the center coordinates as a tuple (x, y).
  - Returns None if the marker is not found.
* Example Call: get_marker_center(corners, marker_id, ids)
"""

def get_marker_center(corners, marker_id, ids):
    index = np.where(ids == marker_id)[0]
    if index.size > 0:
        marker_corners = corners[index[0]].reshape((4, 2))
        center = marker_corners.mean(axis=0)
        return center
    return None

"""
* Function Name: detect_and_estimate_turning_direction
* Input:
  - img_resized (numpy.ndarray): The resized image.
  - corners (numpy.ndarray): The corners of the markers.
  - ids (numpy.ndarray): The IDs of the markers.
* Output:
  - None
* Logic:
  - Checks if marker IDs and stop locations are available.
  - Retrieves the corners of the bot marker based on its ID ('100').
  - Calculates the direction ('N', 'E', 'W', 'S') of the bot marker.
  - Estimates the turning direction ('F', 'R', 'L', 'B') based on the relative position of reference and target markers.
  - Updates the global variables `turnDirection` and `nextStopAngle` accordingly.
* Example Call: detect_and_estimate_turning_direction(img_resized, corners, ids)
"""

def detect_and_estimate_turning_direction(img_resized, corners, ids):
   
    global turnDirection
    global nextStopAngle

    if ids is not None and nextStop is not None and thisStop is not None:
        ids = ids.flatten()
        reference_center = stops_only_dict[thisStop]
        target_center = stops_only_dict[nextStop]
        if thisStop == 'E' and nextStop == 'd':
            reference_center = stops_only_dict[nextStop]
            target_center = stops_only_dict[path_list[current_pointer + 2]]
        if thisStop ==  'A' and nextStop == 'a':
            reference_center = stops_only_dict[nextStop]
            target_center = stops_only_dict[path_list[current_pointer + 2]]
        index = np.where(ids == '100')[0]
        if index.size > 0:
            bot_corners = corners[index[0]].reshape((4, 2))
        else:
            return
        direction = get_marker_direction(bot_corners)
        if reference_center is not None and target_center is not None:
            # Calculate the direction vector from reference to target
            direction_vector = target_center - reference_center
            # Calculate the angle of the direction vector
            angle = np.arctan2(direction_vector[1], direction_vector[0])
            anglePrint = np.arctan2(direction_vector[1], direction_vector[0])
            anglePrint = np.degrees(anglePrint)
            if direction == 'N':
                angle = np.degrees(angle) + 90
            elif direction == 'S':
                angle = np.degrees(angle) + 270
            elif direction == 'W':
                angle = np.degrees(angle) + 180
            elif direction == 'E':
                angle = np.degrees(angle)
            angle = angle % 360
            if angle > 180:
                angle -= 360
            if -45 < angle <= 45:
                turnDirection = "F"
            elif 45 < angle <= 135:
                turnDirection = "R"
            elif -135 < angle <= -45:
                turnDirection = "L"
            else:
                turnDirection = "B"
            nextStopAngle = angle
        else:
            print("Could not find both the reference and the target markers.")

"""
* Function Name: write_lat_lon_to_output_csv
* Input:
  - lat_lon_data (list): A list containing latitude and longitude data.
  - csv_name (str): The name of the CSV file to be created.
* Output: Returns a list representing the coordinate [latitude, longitude] that was written to the CSV file.
* Logic:
  - Extracts latitude and longitude values from the `lat_lon_data` list.
  - Creates a new CSV file named as specified in `csv_name`.
  - Writes headers "lat" and "lon" to the CSV file.
  - Writes the extracted coordinate [latitude, longitude] to the CSV file.
  - Returns the written coordinate.
* Example Call: write_lat_lon_to_output_csv(lat_lon_data, "output_coordinates.csv")
"""

def write_lat_lon_to_output_csv( lat_lon_data, csv_name):
    coordinate = [lat_lon_data[0][0], lat_lon_data[0][1]]
    with open(csv_name, "w", newline='') as file_name:
        csv_writer = csv.writer(file_name)
        csv_writer.writerow(["lat", "lon"])
        csv_writer.writerow(coordinate)
    return coordinate

"""
* Function Name: append_traversed_points_csv
* Input:
  - csv_data (pandas.DataFrame): The CSV data containing the coordinates.
  - target_id (int): The ID of the target.
  - csv_log_name (str): The name of the CSV log file.
* Output: Returns a list representing the coordinates [lat, lon] if successfully appended, None otherwise.
* Logic:
  - Retrieves the coordinates corresponding to the `target_id` from the provided `csv_data`.
  - Checks for duplicate entries in the CSV log file before appending.
  - Appends the coordinates to the specified CSV log file, creating the file if it doesn't exist.
  - Returns the appended coordinates if successful, None otherwise.
* Example Call: append_traversed_points_csv(csv_data, 1, "traversal_log.csv")
"""

def append_traversed_points_csv(csv_data, target_id, csv_log_name):
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

"""
* Function Name: nearestStop
* Input:
  - nearestId (str): The ID of the nearest stop.
* Output: Returns None.
* Logic:
  - Defines a dictionary `stopDict` mapping stop groups to their respective IDs.
  - Finds the top-level stop ID (`thisStop`) based on the given `nearestId`.
  - Sets the global variable `thisStop` to the identified stop ID.
* Example Call: nearestStop('A')
"""

def nearestStop(nearestId):
    stopDict = {
        'A' : ['A'], 'B': ['B'], 'C' : ['C'],  
        'D' : ['D'], 'E' : ['E'], 'a' : ['a'],  
        'b' : ['b'], 'c' : ['c'], 'd' : ['d'], 
        'e' : ['e'], 'f' : ['f'], 'g' : ['g'], 
        'h': ['h'], 'i': ['i'], 'j' : ['j'], 'k' : ['k'],
        'x' : ['x']
    }
    global thisStop
    if nearestId is not None:
        topNearestId = nearestId
        for key, values in stopDict.items():
            if topNearestId in values:
                thisStop = key
                break
    return None

"""
* Function Name: send_to_esp32
* Input: None
* Output: None
* Logic:
  - Retrieves the global variable `turnDirection` and checks the current stop (`thisStop`).
  - Formats a message with separators and encodes it in UTF-8.
  - Sends the formatted message using a UDP socket to the specified IP address and port.
  - Handles special cases for turn direction adjustments.
* Example Call: send_to_esp32()
"""

def send_to_esp32():
    global firstPakageSend
    global turnDirection
    if thisStop is not None:
        if thisStop == 'x' and path_list[current_pointer + 2] == 'A':
            turnDirection = 'R'
        # Format the message with separators
        if len(path_list) > current_pointer + 1:
            message = f"{thisStop}:{nextStop}:{turnDirection}{mask[current_pointer + 1]}"
        else:
            message = f"{thisStop}:{nextStop}:{turnDirection}{mask[current_pointer]}"
        data = message.encode('utf-8')


        # Create a UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Send the UDP packet
            sock.sendto(data, (esp_ip, esp_port))
            # print("Message sent successfully.")
        except Exception as e:
            print(f"Error sending message: {e}")
        finally:
            # Close the socket in a finally block to ensure it gets closed even if an exception occurs
            sock.close()
            firstPakageSend = True

"""
* Function Name: update_path_list_pointer_and_next_stop
* Input: None
* Output: None
* Logic:
  - Retrieves the global variables `current_pointer` and `nextStop`.
  - Checks if the current stop (`thisStop`) is not None and not equal to 'x'.
  - If the current stop matches the next stop in the `path_list`, increments the `current_pointer` and updates the `nextStop` variable.
* Example Call: update_path_list_pointer_and_next_stop()
"""

def update_path_list_pointer_and_next_stop():
    global current_pointer
    global nextStop
    if thisStop is not None and thisStop != 'x':
        if thisStop == path_list[current_pointer + 1]:
            current_pointer = current_pointer + 1
            nextStop = path_list[current_pointer + 1]
    
"""
* Function Name: top_view_tranform
* Input: None
* Output: matrix (numpy.ndarray) - The transformation matrix.
* Logic:
  - Retrieves the global variable `crop_coordinates_list`.
  - Defines source points (`src_pts`) based on the corners of the cropped image.
  - Defines destination points (`dst_pts`) for the desired top view.
  - Calculates the perspective transformation matrix (`matrix`) using OpenCV's `getPerspectiveTransform` function.
  - Prints the matrix and returns it.
* Example Call: top_view_tranform()
"""  
     
def top_view_tranform():
    src_pts = np.float32([crop_coordinates_list[3], crop_coordinates_list[2], crop_coordinates_list[1], crop_coordinates_list[0]])
    dst_pts = np.float32([[0,0], [800, 0], [0, 800], [800, 800]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return matrix

"""
* Function Name: qgis_tranform
* Input: tranformed_image (numpy.ndarray) - The image to be transformed.
* Output: numpy.ndarray - The transformation matrix.
* Raises: None
* Logic:
  - Calls the `aruco_dict_ret` function to get the corners and ids of Aruco markers in the transformed image.
  - If ids are not found, returns None.
  - Defines a list of ids to find in the Aruco markers.
  - Calculates the average position of specified Aruco markers using their ids and weights.
  - Defines source (`pts_src`) and destination (`pts_dst`) points for perspective transformation based on the calculated averages.
  - Uses OpenCV's `getPerspectiveTransform` function to obtain the transformation matrix.
  - Returns the transformation matrix.
* Example Call: qgis_tranform(tranformed_image)
"""

def qgis_tranform(tranformed_image):
    corners ,ids= aruco_dict_ret(tranformed_image)
    
    if ids is None:
        return None
    ids_to_find = [22, 49, 25, 51, 52, 42, 36, 10, 8, 29, 11, 13]
    aruco_position = {}
    for id_ in ids_to_find:
        index = (ids == id_).nonzero()[0]
        if len(index) == 0:
            return None
        id_corners = corners[index[0]][0]
        center = np.mean(id_corners, axis=0)
        aruco_position[id_] = center
    weights = np.array([0.6, 0.6, 0.7], dtype=np.float32)
    b = np.array(np.average([aruco_position[22], aruco_position[49], aruco_position[25]], axis=0, weights=weights), dtype=np.float32)
    weights = np.array([0.6, 0.5, 0.9], dtype=np.float32)
    d = np.array(np.average([aruco_position[51], aruco_position[52],aruco_position[42]], axis=0, weights=weights), dtype=np.float32)
    weights = np.array([0.5, 0.65, 0.05], dtype=np.float32)
    e = np.array(np.average([aruco_position[36], aruco_position[10],aruco_position[8]], axis=0, weights=weights), dtype=np.float32)
    weights = np.array([0.5, 0.525, 0.05], dtype=np.float32)
    g = np.array(np.average([aruco_position[29], aruco_position[11],aruco_position[13]], axis=0, weights=weights), dtype=np.float32)
    # Define four points in the original image
    pts_src = np.array([[39.61320127,-74.36285562], [39.61375159,-74.36267339], [39.61337050,-74.36100523], [39.61289738,-74.36115948]], np.float32)
    pts_dst = np.array([b, d, e, g], np.float32)
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    return matrix
    
"""
* Function Name: main
* Input: None
* Output: None
* Logic:
  - Initializes variables and reads CSV data from a file.
  - Enters an infinite loop to continuously read frames from the camera.
  - Detects Aruco markers in the frames, processes them, determines turning direction, and displays the resized image with bounding contours.
  - Continues to run until the 'q' key is pressed.
  - Uses various helper functions such as `fillArucoDict_and_cropList`, `read_event_dict`, `floyd_warshall`, `reconstruct_combined_path`, `top_view_tranform`, `qgis_tranform`, `resizeImage_and_nearestAruro_botmarker`, `arena_to_qgis_mapping`, `detect_and_estimate_turning_direction`, `write_lat_lon_to_output_csv`, `append_traversed_points_csv`, `nearestStop`, `update_path_list_pointer_and_next_stop`, and `send_to_esp32`.
* Example Call: main()
"""

def main():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    fetched = False
    inputFile = "lat_long.csv"
    csv_data = pd.read_csv(inputFile)
    global lastId
    global path_list
    global nextStop
    while True:
        ret, frame = cap.read()
        if fetched is False: 
            corners, ids = aruco_dict_ret(frame)
            condition = fillArucoDict_and_cropList(frame, corners, ids)
            if condition is True:
                read_event_dict()
                _, next_vertex = floyd_warshall(Adj_matrix)
                combined_dict = reconstruct_combined_path(event_priority, id_mapping_dictionary, next_vertex)
                path_list = [id_mapping_dictionary[key] for key in combined_dict]
                fetched = True
                nextStop = path_list[current_pointer + 1]
                top_view_image_cropped = cv2.warpPerspective(frame, top_view_tranform_matrix, (800, 800))
                qgis_tranform_matrix = qgis_tranform(top_view_image_cropped)
            else:
                continue
        # Handle frame reading error.
        if not ret:
            print("Error reading frame from the camera.")
        # Process the frame to resize and find nearest marker.
        corners, ids = aruco_dict_ret(frame)
        new_values = [['A'], ['B'], ['C'], ['D'], ['E'], ['a'],  ['b'], ['c'],  ['d'], ['e'], ['f'],  ['g'], ['h'],  ['i'],  ['j'],  ['k'], ['x']]
        ids = ids.tolist()
        # Append new values to the existing array
        ids += new_values
        ids = np.array(ids)
        img_resized, nearestId, _, nearestAruco_Id = resizeImage_and_nearestAruro_botmarker(frame, corners, ids)
        img_to_show = cv2.warpPerspective(frame, top_view_tranform_matrix, (800, 800))
        lat_lon = arena_to_qgis_mapping(img_to_show, qgis_tranform_matrix)
        if lat_lon is None:
            continue
        outputFile = "output.csv"
        write_lat_lon_to_output_csv(lat_lon, outputFile)
        # Determine turning direction
        if len(corners) > 0 and ids is not None:
            detect_and_estimate_turning_direction(img_resized, corners, ids)
        if nearestId is not None:
            lastId = nearestId
        if lastId is not None:
            nearestStop(lastId)
            logfile = "task_4b.csv"
            append_traversed_points_csv(csv_data, nearestAruco_Id, logfile)
            
        update_path_list_pointer_and_next_stop()
        # Send data to ESP32.
        send_to_esp32()  
        # Display the resized image.
        cv2.namedWindow('Image with Bounding Contours', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image with Bounding Contours', 960, 960)
        cv2.moveWindow('Image with Bounding Contours', 0, 0)
        cv2.imshow('Image with Bounding Contours', img_to_show)
        # Exit loop if 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release camera and close all OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()