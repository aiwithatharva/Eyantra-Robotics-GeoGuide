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
# Filename:			Task_5a_FinalRun.py


####################### IMPORT MODULES #######################

import cv2
import cv2.aruco as aruco
import numpy as np
import pandas as pd
import csv
import time
import socket

# WiFi credentials
ssid = "abc"
password = "11111111"

# ESP32 IP and port
esp_ip = "192.168.70.224"
esp_port = 4210
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Global variables
thisStop = None
nextStop = None
turnDirection = None
lastId = None

# Path list and pointer
current_pointer = 0
path_list = []
mask = []


# Lists for coordinates and ArUco markers
crop_coordinates_list = []
aruco_marker_dict = {}

inf = float('inf')
Adj_matrix = np.load('filtered_adjacency_matrix.npy')
id_mapping_dictionary = {    
    0: 10,
    1: 12,
    2: 15,
    3: 19,
    4: 11,
    5: 95,
    6: 22,
    7: 23,
    8: 24,
    9: 27,
    10: 96,
    11: 97,
    12: 32,
    13: 98,
    14: 39,
    15: 99,
    16: 50,
    17: 51,
    18: 53
}

event_location_dict = { 'A' : '95', 'B' : '96', 'C' : '97', 'D' : '98', 'E' : '99' }

event_priority = {
    'Fire': {'priority': 1, 'position': None},
    'Destroyed Buildings': {'priority': 2, 'position': None}, 
    'Humanitarian Aid and rehabilitation': {'priority': 3, 'position': None},  
    'Military Vehicles': {'priority': 4, 'position': None},
    'Combat': {'priority': 5, 'position': None}  
}

def aruco_dict_ret(frame):
    """
    Detects ArUco markers in the input frame and returns their corners and IDs.

    This function identifies ArUco markers present in the provided frame using image processing techniques. 
    It returns the corners of each detected marker, which are essential for various applications such as 
    spatial analysis and augmented reality, along with their unique IDs. The corners are the coordinates 
    of the four corners of the square marker, which can be used to determine the marker's orientation 
    and position in the frame.

    Args:
        frame (np.array): The input frame as a numpy array, typically captured from a video feed, 
                          in which ArUco markers are to be detected.

    Returns:
        tuple: A tuple containing two elements:
            - corners: A list of the coordinates for the corners of each detected ArUco marker.
            - ids: An array of the detected markers' IDs.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.convertScaleAbs(gray, alpha=0.3, beta=1)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(result, aruco_dict, parameters=parameters)
    return (corners, ids)

    
def read_event_dict():
    """
    Reads event information from 'Event_output.csv' and updates the global event_priority dictionary.

    This function opens 'Event_output.csv', reads the event data (which includes event types and their corresponding locations),
    and updates the event_priority dictionary with the current location (position) for each event type. The priority of the events
    remains unchanged. The 'Event_output.csv' file is expected to contain event types and their new positions, which are used to
    update the global event_priority dictionary, ensuring that the system is aware of the current event locations for decision-making
    processes.

    Parameters:
    None

    Returns:
    None

    Updates:
    - event_priority: A global dictionary with keys representing different event types (e.g., 'fire', 'destroyed_buildings') and
      values being dictionaries with 'priority' (an integer indicating the urgency of the event) and 'position' (updated with the
      current location of the event based on the CSV file).

    Raises:
    - FileNotFoundError: If 'Event_output.csv' does not exist in the expected directory.
    - KeyError: If the CSV contains event types not predefined in the event_priority dictionary.
    """
    global event_priority
    with open('Event_output.csv') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        event_dict = next(csv_reader, None)
        if event_dict:
            for location in event_dict:
                event = event_dict[location]
                if event != 'None':
                    event_priority[event] = {'priority': event_priority[event]['priority'], 'position': event_location_dict[location]}
def get_priority(event):
    return event_priority[event]['priority']

def get_position(event):
    return event_priority[event]['position']


def floyd_warshall(Adj_matrix):
    n = len(Adj_matrix)
    dist = [[float('inf') for _ in range(n)] for _ in range(n)]
    next_vertex = [[None for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            dist[i][j] = Adj_matrix[i][j]
            if Adj_matrix[i][j] != float('inf') and i != j:
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

def reconstruct_path(start, end, next_vertex):
    """
    Reconstructs the shortest path between start and end vertices and a mask indicating the visited vertices.

    This function iterates through a mapping of each vertex to the next vertex on the shortest path from 
    the start to that vertex, effectively reconstructing the shortest path from the end vertex back to the 
    start. It also generates a mask that indicates which vertices were visited during the path reconstruction. 
    This is useful for visualizing the path or analyzing the coverage of the traversal algorithm.

    Args:
        start: The starting vertex in the path.
        end: The ending vertex in the path.
        next_vertex (dict): A dictionary mapping each vertex to the next vertex on the shortest path 
                            from the start vertex to that vertex.

    Returns:
        tuple: A tuple containing two elements:
            - path: A list of vertices representing the shortest path from start to end, inclusive of both.
            - mask: A list or array indicating the visited vertices, where each element corresponds to a 
                    vertex and its value indicates whether the vertex was visited (1) or not (0) during 
                    the path reconstruction.
    """

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


def reconstruct_combined_path(event_priority, your_dictionary, next_vertex):
    """
    Generates an optimized path through events based on priority and shortest paths.

    This function creates a path that navigates through events in order of priority, using the shortest 
    paths between them. It leverages event_priority for order, your_dictionary for event details, and 
    next_vertex for mapping shortest paths. The aim is to efficiently visit all prioritized events.

    Args:
        event_priority (list): List of event IDs in priority order.
        your_dictionary (dict): Dictionary with events' details.
        next_vertex (dict): Mapping of each vertex to the next on the shortest path.

    Returns:
        list: Optimized sequence of vertices forming the combined path through prioritized events.
    """
    global mask
    start_position = 23
    
    # Find the first event with a non-None position
    for event, info in event_priority.items():
        if info['position'] is not None:
            event1 = event
            break
    else:
        print("No event with a non-None position found.")
        return None
    
    intermediate_position = int(get_position(event1))
    end_position = None
    
    # Find the second event with a non-None position
    for event, info in event_priority.items():
        if info['position'] is not None and event != event1:
            event2 = event
            end_position = int(info['position'])
            break
    else:
        print("No second event with a non-None position found.")
        return None
    
    start = next(key for key, value in your_dictionary.items() if value == start_position)
    intermediate = next(key for key, value in your_dictionary.items() if value == intermediate_position)
    end = next(key for key, value in your_dictionary.items() if value == end_position)
    
    path1, mask1 = reconstruct_path(start, intermediate, next_vertex)
    path2 , mask2= reconstruct_path(intermediate, end, next_vertex)
    
    if path1 is None or path2 is None:
        return None
    
    # Determine the order of paths based on event priorities
    if get_priority(event1) <= get_priority(event2):
        path2.pop(0)
        mask2.pop(0)
    else:
        path1.pop(0)
        mask1.pop(0)
    path3, mask3 = reconstruct_path(end, start, next_vertex)
    path3.pop(0)
    mask3.pop(0)
    mask3.append(0)
    path3.append(0)
    combined_path = path1 + path2+path3
    mask = mask1 + mask2 + mask3
    return combined_path
   

def fillArucoDict_and_cropList(frame, corners, ids):
    """
    Updates an ArUco dictionary and generates cropped images for detected markers.

    Args:
        frame (np.array): Image frame with ArUco markers.
        corners (list): Detected marker corners.
        ids (np.array): Marker IDs.

    Returns:
        tuple: Updated dictionary with marker IDs and corners, list of cropped marker images.
    """

    global crop_coordinates_list
    if ids is None:
        return False
    if len(ids) < 52:
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

    weights = np.array([0.5, 0.5], dtype=np.float32)
    aruco_marker_dict[99] = np.array(np.average([aruco_marker_dict[53], aruco_marker_dict[48]], axis=0, weights=weights), dtype=np.float32)
    weights = np.array([0.2, 0.8], dtype=np.float32)
    aruco_marker_dict[98] = np.array(np.average([aruco_marker_dict[50], aruco_marker_dict[34]], axis=0, weights=weights), dtype=np.float32)
    weights = np.array([0.5, 0.5], dtype=np.float32)
    aruco_marker_dict[97] = np.array(np.average([aruco_marker_dict[31], aruco_marker_dict[30]], axis=0, weights=weights), dtype=np.float32)
    weights = np.array([0.5, 0.5], dtype=np.float32)
    aruco_marker_dict[96] = np.array(np.average([aruco_marker_dict[29], aruco_marker_dict[28]], axis=0, weights=weights), dtype=np.float32)
    weights = np.array([0.8, 0.2], dtype=np.float32)
    aruco_marker_dict[95] = np.array(np.average([aruco_marker_dict[21], aruco_marker_dict[24]], axis=0, weights=weights), dtype=np.float32)
    weights = np.array([0.4, 0.6], dtype=np.float32)
    aruco_marker_dict[94] = np.array(np.average([aruco_marker_dict[7], aruco_marker_dict[23]], axis=0, weights=weights), dtype=np.float32)
    return True


 
def resizeImage_and_nearestAruro_botmarker(frame, corners, ids):
    """
    Resizes the image frame and identifies the nearest ArUco marker to the bot.

    This function adjusts the frame size for optimal processing and uses the corners and ids of detected 
    ArUco markers to determine which marker is closest to the robotic entity, assuming the bot's position 
    is predefined or determined by another component of the system.

    Args:
        frame (np.array): The original frame to be resized.
        corners (list): Coordinates of the ArUco marker corners.
        ids (np.array): IDs of the detected ArUco markers.

    Returns:
        tuple: Resized frame and ID of the nearest ArUco marker to the bot.
    """

    
    nearest_id = None
    idZero_corners = None

    if ids is not None:
        target_index = np.where(ids == 100)[0]
        if len(target_index) > 0:
            idZero_corners = corners[target_index[0]][0]
            target_center = np.mean(idZero_corners, axis=0)
            # Compute distances to other markers
            custom_range = list(range(4, 55)) + list(range(94, 100))
            id_distance_pairs = [(id, np.linalg.norm(target_center - aruco_marker_dict[id])) for id in custom_range]      
            sorted_pairs = sorted(id_distance_pairs, key=lambda pair: pair[1])
            distance_threshold = 80

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

def get_marker_direction(corners):
    """
    Determines the direction of a marker based on its corner points.

    The direction is calculated using the angle of the line formed by the top left and top right corners of the marker.

    Parameters:
    corners (numpy.ndarray): The corner points of the marker.

    Returns:
    Direction (str): The direction of the marker, either 'N', 'E', 'S', or 'W'.
    """
    
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

def get_marker_center(corners, marker_id, ids):
    """
    Computes the center point of a specified ArUco marker identified by its ID.

    This function locates the specific marker based on the provided marker_id and ids list, then calculates 
    the geometric center of the marker using its corner coordinates. This center point can be used for 
    positioning, alignment, or as a reference point in various applications.

    Args:
        corners (list): List of corner coordinates for all detected ArUco markers.
        marker_id (int): The ID of the marker for which to find the center.
        ids (list): List of all detected marker IDs.

    Returns:
        tuple: The (x, y) coordinates of the marker's center, or (None, None) if the marker is not found.
    """
    index = np.where(ids == marker_id)[0]
    if index.size > 0:
        marker_corners = corners[index[0]].reshape((4, 2))
        center = marker_corners.mean(axis=0)
        return center
    return None

def detect_and_estimate_turning_direction(img_resized, corners, ids):
    """
    Detects markers in a resized image and estimates the turning direction based on their orientation.

    Processes the resized image to identify ArUco markers using their corners and IDs, then estimates 
    the turning direction required to align with or navigate towards these markers. The direction is 
    determined based on the relative orientation of the markers to a predefined point or the camera's 
    current orientation.

    Args:
        img_resized (np.array): The resized image for processing.
        corners (list): Corner coordinates of detected ArUco markers.
        ids (list): IDs of the detected ArUco markers.

    Returns:
        None
    """
    global turnDirection

    if ids is not None and nextStop is not None:
        ids = ids.flatten()
        reference_center = get_marker_center(corners, 100, ids)
        target_center = aruco_marker_dict[nextStop]
        if int(thisStop) == 98 and int(nextStop) == 50 and path_list[int(current_pointer + 2)] == 51:
            target_center = aruco_marker_dict[51]
        elif int(thisStop) == 98 and int(nextStop) == 50 and path_list[int(current_pointer + 2)] == 22:
            target_center = aruco_marker_dict[22]
        index = np.where(ids == 100)[0]
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
        else:
            print("Could not find both the reference and the target markers.")


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
        '23' : [94], '24': [23, 24], '22' : [22],  
        '50' : [50, 49], '51' : [51], '53' : [53],  
        '99' : [99], '10' : [36, 10, 8], '12' : [12, 30], 
        '11' : [29, 11, 13], '15' : [15], '19' : [20, 19], 
        '95': [95], '27': [27, 23], '98' : [98], '32' : [33, 32],
        '97' : [97], '39' : [39, 35],'96': [96]
    }
    global thisStop
    if nearestId is not None:
        topNearestId = nearestId
        for key, values in stopDict.items():
            if topNearestId in values:
                thisStop = key
                break
    return None


def send_to_esp32():
    """
    Sends the current stop and direction information to the ESP32 via UDP.
    This function constructs a message containing the current stop and direction, encodes it, and sends it to the ESP32 device 
    specified by the global variables esp_ip and esp_port. It only sends the data if both thisStop and direction are not None.

    Returns:
    None
    """
    if thisStop is not None and turnDirection is not None:
        # Format the message with separators
        if len(path_list) > current_pointer + 1:
            message = f"{int(thisStop):02d}:{nextStop:02d}:{turnDirection}{mask[current_pointer + 1]}"
        else:
            message = f"{int(thisStop):02d}:{nextStop:02d}:{turnDirection}{mask[current_pointer]}"

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

def update_path_list_pointer_and_next_stop():
    global current_pointer
    global nextStop
    if thisStop is not None:
        if int(thisStop) == path_list[current_pointer + 1]:
            current_pointer = current_pointer + 1
            nextStop = path_list[current_pointer + 1]
    
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
    global path_list
    while True:
        ret, frame = cap.read()
        if fetched is False:
            # Initial detection of ArUco markers.
            corners, ids = aruco_dict_ret(frame)
            condition = fillArucoDict_and_cropList(frame, corners, ids)
            if condition is True:
                read_event_dict()
                _, next_vertex = floyd_warshall(Adj_matrix)
                combined_dict = reconstruct_combined_path(event_priority, id_mapping_dictionary, next_vertex)
                path_list = [id_mapping_dictionary[key] for key in combined_dict]
                fetched = True
            else:
                continue
        # Handle frame reading error.
        if not ret:
            print("Error reading frame from the camera.")
        # Process the frame to resize and find nearest marker.
        corners, ids = aruco_dict_ret(frame)
        new_values = [[99], [98], [97], [96], [95], [94]]
        ids = ids.tolist()
        # Append new values to the existing array
        ids += new_values
        ids = np.array(ids)
        img_resized, nearestId, _ = resizeImage_and_nearestAruro_botmarker(frame, corners, ids)
        if img_resized is None:
            continue
        # Determine turning direction
        if len(corners) > 0 and ids is not None:
            detect_and_estimate_turning_direction(img_resized, corners, ids)
        if nearestId is not None:
            lastId = nearestId
        if lastId is not None:
            nearestStop(lastId)
            outputFile = "output.csv"
            find_and_write_nearest_csv_lat_lon(csv_data,lastId, outputFile)
            logfile = "task_4b.csv"
            append_traversed_points_csv(csv_data, lastId, logfile)
            
        update_path_list_pointer_and_next_stop()
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