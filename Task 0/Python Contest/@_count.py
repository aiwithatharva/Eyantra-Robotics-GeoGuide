'''
# Team ID:          3303
# Theme:            GeoGuide
# Author List:      Atharva Satish Attarde, Nachiket Ganesh Apte, Ashutosh Anil Dongre, Prachit Suresh Deshinge
# Filename:         @_count.py
# Functions:        -
# Global variables: None
'''
def main():
    '''    Purpose:
    ---
    Asks the user to input the number of test cases. Then ask senternces starting with @ then count length of each word seperated by comma.
    
    Input Arguments:
    ---
    None
    
    Returns:
    ---
    None
    '''
    number_of_test = int(input())
    for i in range(number_of_test):
        text = input()
        x = text.split(" ")
        x[0] = x[0].replace('@', '')
        for i in x:
            if i == x[-1]:
                print(len(i))
            else:
                print(len(i), end=',')
                
            
if __name__ == "__main__":
    main()