'''
# Team ID:          3303
# Theme:            GeoGuide
# Author List:      Atharva Satish Attarde, Nachiket Ganesh Apte, Ashutosh Anil Dongre, Prachit Suresh Deshinge
# Filename:         sliceList.py
# Functions:        main
# Global variables: None
'''
def main():
    
    '''    Purpose:
    ---
    Asks the user to input the number of test cases . Then ask number of inputs  , then list of numbers and return solution as given in problem. 
    
    Input Arguments:
    ---
    None
    
    Returns:
    ---
    None
    '''
    number_of_test = int(input())
    for i in range(number_of_test):
        lenght_list = int(input())
        mylist_list = input().split()
        mylist_list = list(map(int, mylist_list))
        rever_list = mylist_list[::-1]
        _3list = []
        _5list = []
        sum = 0
        for i in range(len(mylist_list)):
            if i % 3 == 0 and i != 0:
                _3list.append(mylist_list[i] + 3)
            if i % 5 == 0 and i != 0:
                _5list.append(mylist_list[i] - 7)
            if i >= 3 and i <= 7:
                sum += mylist_list[i]
        for i in range(len(rever_list)):
            if i == len(rever_list) - 1:
                print(rever_list[i])
            else:
                print(rever_list[i], end=' ')
        for i in range(len(_3list)):
            if i == len(_3list) - 1:
                print(_3list[i])
            else:
                print(_3list[i], end=' ')
        for i in range(len(_5list)):
            if i == len(_5list) -1:
                print(_5list[i])
            else:
                print(_5list[i], end=' ')
        print(sum)
        
if __name__ == "__main__":
    main()