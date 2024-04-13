'''
# Team ID:          3303
# Theme:            GeoGuide
# Author List:      Atharva Satish Attarde, Nachiket Ganesh Apte, Ashutosh Anil Dongre, Prachit Suresh Deshinge
# Filename:         topper.py
# Functions:        main
# Global variables: None
'''
def main():
    '''    Purpose:
    ---
    Asks the user to input the number of test cases. Then ask data of name and marks , return the topper.
    Input Arguments:
    ---
    None
    
    Returns:
    ---
    None
    '''
    number_of_test = int(input())
    for i in range(number_of_test):
        number_of_data = int(input())
        my_dict = {}
        for data in range(number_of_data):
            x = input().split()
            my_dict[x[0]] = float(x[1])
        max_marks = max(my_dict.values())
        final_list = []
        for key, val in my_dict.items():
            if max_marks == val:
                final_list.append(key)
        final_list.sort()
        print(final_list[0])
    
if __name__ == "__main__":
    main()