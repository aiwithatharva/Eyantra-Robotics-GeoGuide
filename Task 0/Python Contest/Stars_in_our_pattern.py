'''
# Team ID:          3303
# Theme:            GeoGuide
# Author List:      Atharva Satish Attarde, Nachiket Ganesh Apte, Ashutosh Anil Dongre, Prachit Suresh Deshinge
# Filename:         Stars_in_our_pattern.py
# Functions:        -
# Global variables: None
'''

def main():
    '''    Purpose:
    ---
    Asks the user to input the number of test cases. Then ask integer n and print pattern as per the requirement.
    
    Input Arguments:
    ---
    None
    
    Returns:
    ---
    None
    '''
    number_of_test = int(input())
    for k in range(number_of_test):
        n = int(input())
        for j in range(n, 0, -1):
            for i in range(j):
                if (i + 1) % 5 == 0 and i == j - 1:
                    print('#')
                elif (i + 1) % 5 == 0:
                    print('#',end='')
                elif i == j - 1:
                    print('*')
                else:
                    print('*',end='')
                
if __name__ == "__main__":
    main()