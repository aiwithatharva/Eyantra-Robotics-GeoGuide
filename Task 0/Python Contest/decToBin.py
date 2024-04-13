'''
# Team ID:          3303
# Theme:            GeoGuide
# Author List:      Atharva Satish Attarde, Nachiket Ganesh Apte, Ashutosh Anil Dongre, Prachit Suresh Deshinge
# Filename:         decToBin.py
# Functions:        -
# Global variables: None
'''


def dec_to_binary(n):
    '''    Purpose:
    ---
    A recursive funtion to convert a decimal number n and return binary equivalent 
    
    Input Arguments:
    ---
    Decimal number, infunction multiplier, binary sum.
    
    Returns:
    ---
    Binary number
    '''
    global offset
    global bin_num
    if n // 2 == 0:
        bin_num += (n % 2) * offset
        return bin_num
    bin_num += (n % 2) * offset
    offset *= 10
    return dec_to_binary(n // 2)




def main():
    '''    Purpose:
    ---
    Asks the user to input the number of test cases. Then ask decimal number
    
    Input Arguments:
    ---
    None
    
    Returns:
    ---
    None
    '''    
    global offset

    global bin_num

    test_cases = int(input())

    for case in range(1, test_cases + 1):
        offset = 1
        bin_num = 0
        n = int(input())
        bin_num = dec_to_binary(n)
        print("{:08}".format(bin_num))
        
if __name__ == "__main__":
    main()