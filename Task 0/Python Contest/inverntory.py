'''
# Team ID:          3303
# Theme:            GeoGuide
# Author List:      Atharva Satish Attarde, Nachiket Ganesh Apte, Ashutosh Anil Dongre, Prachit Suresh Deshinge
# Filename:         inventory.py
# Functions:        main
# Global variables: None
'''
def main():
    '''    Purpose:
    ---
    Asks the user to input the number of test cases . Then ask num of inital items and their quantities , then ask for operation to perform on the inventory. Print on screen as per required in the problem.
    
    Input Arguments:
    ---
    None
    
    Returns:
    ---
    None
    '''
    number_of_test = int(input())
    for i in range(number_of_test):
        my_dict = {}
        init_number = int(input())
        for i in range(init_number):
            x = input().split()
            my_dict[x[0]] = int(x[1])
        num_operation = int(input())
        
        for i in range(num_operation):
            x = input().split()
            if x[0] == "ADD":
                if x[1] not in my_dict:
                    my_dict[x[1]] = int(x[2])
                    print("ADDED Item {}".format(x[1]))
                elif x[1] in my_dict:
                    my_dict[x[1]] = my_dict[x[1]] + int(x[2])
                    print("UPDATED Item {}".format(x[1]))
            elif x[0] == "DELETE":
                if x[1] not in my_dict:
                    print("Item {} does not exist".format(x[1]))
                elif x[1] in my_dict and int(x[2]) > my_dict[x[1]]:  
                    print("Item {} could not be DELETED".format(x[1]))
                else:
                    my_dict[x[1]] = my_dict[x[1]] - int(x[2])
                    print("DELETED Item {}".format(x[1]))
        sum = 0
        for i in my_dict.values():
            sum += i
        print("Total Items in Inventory: {}".format(sum))
        
if __name__ == "__main__":
    main()