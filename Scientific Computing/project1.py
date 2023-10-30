"""
Code for Scientific Computation Project 1
Please add college id here
CID: 02045099
"""


#===== Code for Part 1=====#
def part1(Xin,istar):
    """
    Sort list of integers in non-decreasing order
    Input:
    Xin: List of N integers
    istar: integer between 0 and N-1 (0<=istar<=N-1)
    Output:
    X: Sorted list
    """ 
    X = Xin.copy() 
    for i,x in enumerate(X[1:],1):
        if i<=istar:
            ind = 0
            for j in range(i-1,-1,-1):
                if x>=X[j]:
                    ind = j+1
                    break                   
        else:
            a = 0
            b = i-1
            while a <= b:
                c = (a+b) // 2
                if X[c] < x:
                    a = c + 1
                else:
                    b = c - 1
            ind = a
        
        X[ind+1:i+1] = X[ind:i]
        X[ind] = x

    return X


def part1_time(N, n):
    """Examine dependence of walltimes of part1 function on N and istar
        You may modify the input/output as needed.
    """

    #Add code here for part 1, question 2
    import time
    import numpy as np
    import matplotlib.pyplot as plt

    def avgd_sample(Xin, coef,  n):
        #Function to take n number of samples and return the mean of them
        times = []
        for i in range(n):
            t1 = time()
            part1(Xin, coef * (len(Xin) - 1))
            t2 = time()
            times.append(t2 - t1)
        return sum(times)/n
    
    #Xin = np.random.randint(1, 2*N, N)
    #plot1 - fixed istar = 0, N-1 and 2 values in between, with lots of values being simulated, taking means
    x = np.arange(10, 1000, 10)
    t_0 = []
    t_5 = []
    t_75 = []
    t_1 = []
    for length in x:
        Xin = np.random.randint(1, 2*length, length)
        t_0.append(avgd_sample(Xin, 0, 100))

    for length in x:
        Xin = np.random.randint(1, 2*length, length)
        t_5.append(avgd_sample(Xin, 0.5, 100))

    for length in x:
        Xin = np.random.randint(1, 2*length, length)
        t_75.append(avgd_sample(Xin, 0.75, 100))

    for length in x:
        Xin = np.random.randint(1, 2*length, length)
        t_1.append(avgd_sample(Xin, 1, 100))

    plt.plot(x, t_0)
    plt.plot(x, t_5)
    plt.plot(x, t_75)
    plt.plot(x, t_1)


    return None #Modify if needed



#===== Code for Part 2=====#

def part2(S,T,m):
    """Find locations in S of all length-m sequences in T
    Input:
    S,T: length-n and length-l gene sequences provided as strings

    Output:
    L: A list of lists where L[i] is a list containing all locations 
    in S where the length-m sequence starting at T[i] can be found.
   """
    #Size parameters
    n = len(S) 
    l = len(T) 
    
    L = [[] for i in range(l-m+1)] #use/discard as needed

    #Add code here for part 2, question 1
    X = base_4_converter(S)
    Y = base_4_converter(T)
    

    q = 1871
    hash_T = [heval(Y[i:i + m], 4, q) for i in range(l - m + 1)]
    hash_i = heval(X[:m], 4, q)


    for i, value in enumerate(hash_T):
        if hash_i == value:
            if S[:m] == T[i:i + m]:
                L[i].append(0)

    for i in range(1, n - m + 1):
        hash_i = (4 * hash_i - int(X[i]) * ((4 ** m) % q) + int(X[i - 1 + m])) % q
        for j, value in  enumerate(hash_T):
            if hash_i == value:
                if S[i:i + m] == T[j:j + m]:
                    L[j].append(i)
    

    return L


def base_4_converter(S):
    #Function to convert character string into base 4 numbers
    values = {}
    values['A'] = 0
    values['T'] = 1
    values['C'] = 2
    values['G'] = 3

    L = []
    for letter in S:
        L.append(values[letter])

    return L


def heval(L, Base, q):
    f = 0
    for l in L[:-1]:
        f = Base * (l + f)
    
    h = (f + (L[-1])) % q
    return h


if __name__=='__main__':
    #Small example for part 2
    S = 'ATCGTACTAGTTATCGT'
    T = 'ATCGT'
    m = 3
    out = part2(S,T,m)

    #Large gene sequence from which S and T test sequences can be constructed
    infile = open("test_sequence.txt") #file from lab 3
    sequence = infile.read()
    infile.close()
