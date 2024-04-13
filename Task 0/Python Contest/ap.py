'''
# Team ID:          3303
# Theme:            GeoGuide
# Author List:      Atharva Satish Attarde, Nachiket Ganesh Apte, Ashutosh Anil Dongre, Prachit Suresh Deshinge
# Filename:         @_count.py
# Functions:        -
# Global variables: None
'''
# Import reduce module
from functools import reduce
# Function to generate the A.P. series
def generate_AP(a1, d, n):
    count = 0
    AP_series = [] 
    while(1):
        AP_series.append(a1)
        a1 = a1 + d
        count = count + 1
        if count == n:
            return AP_series
# Main function
if __name__ == '__main__':
    # take the T (test_cases) input
    test_cases = int(input())
    # Write the code here to take the a1, d and n values
    a1_d_n = input()
    my_inputs = a1_d_n.split()
    a1 = int(my_inputs[0])
    d = int(my_inputs[1])
    n = int(my_inputs[2])
    # Once you have all 3 values, call the generate_AP function to find A.P. series and print it
    AP_series = generate_AP(a1, d, n)
    for i in AP_series:
        if i == AP_series[-1]:
            print(i)
        else:
            print(i, end=' ')
    # Using lambda and map functions, find squares of terms in AP series and print it
    sqr_AP_series = list(map(lambda x:x*x, AP_series))
    for i in sqr_AP_series:
        if i == sqr_AP_series[-1]:
            print(i)
        else:
            print(i, end=' ')
    # Using lambda and reduce functions, find sum of squares of terms in AP series and print it
    sum_sqr_AP_series = reduce((lambda x, y: x + y), sqr_AP_series)
    print(sum_sqr_AP_series)