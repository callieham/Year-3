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


    #Add code here for part 1, question 2
def avgd_sample(N, coef,  n):
    #Function to take n number of samples of list length N with coef such that istar = coef(N-1) and return the mean of them
    import numpy as np
    import time
    #initialise timer as zero
    timer = 0
    #loop n times and each time, generate list, taker initial time, perform function, take second time, add the difference to the timer
    for i in range(n):
        Xin = np.random.randint(1, 2*N, N)
        t1 = time.time()
        part1(Xin, int(coef * (N - 1)))
        t2 = time.time()
        timer += t2 - t1
    #return the average of the times
    return timer/n


def part1_time(n):
    """Examine dependence of walltimes of part1 function on N and istar
        You may modify the input/output as needed.
    """

    #Add code here for part 1, question 2
    import numpy as np
    import matplotlib.pyplot as plt
    
    #set values for lengths of the lists
    x = np.arange(10, 1000, 10)
    t_0 = []
    t_5 = []
    t_1 = []
    #loop using my sampler function at the difference lengths with different coef values
    for length in x:
        t_0.append(avgd_sample(length, 0, n))
        t_5.append(avgd_sample(length, 0.5, n))
        t_1.append(avgd_sample(length, 1, n))

    t_100 = []
    t_500 = []
    t_1000 = []
    #set values for coefs
    xx = np.arange(0, 1, 0.01)

    #loop using smapler function for different coefs
    for coef in xx:
        t_100.append(avgd_sample(100, coef, n))
        t_500.append(avgd_sample(500, coef, n))
        t_1000.append(avgd_sample(1000, coef, n))


    #initialise plots
    fig, ax = plt.subplots(2, 2)
    fig.tight_layout(w_pad=2.5, h_pad=3.2)
    #plot various lengths with 3 values of coef
    ax[0, 0].set_title('Figure 1')
    ax[0, 0].plot(x, t_0, label = 'istar = 0')
    ax[0, 0].plot(x, t_5, label = 'istar = (N-1)//2')
    ax[0, 0].plot(x, t_1, label = 'istar = N-1')
    ax[0, 0].set_xlabel("N")
    ax[0, 0].set_ylabel("Wall Time (s)")
    ax[0, 0].legend(loc = "upper left")
    #plot xlogx against times for istar = 0
    ax[0, 1].set_title('Figure 2')
    ax[0, 1].plot(x*np.log(x), t_0, label = 'xstar = 0')
    ax[0, 1].set_xlabel("Nlog(N)")
    ax[0, 1].set_ylabel("Wall Time (s)")
    ax[0, 1].legend(loc = 'upper left')
    #plot x^2 against times for istar = N-1
    ax[1, 0].set_title('Figure 3')
    ax[1, 0].plot(x**2, t_1, label = 'istar = N-1')
    ax[1, 0].set_xlabel("N^2")
    ax[1, 0].set_ylabel("Wall Time (s)")
    ax[1, 0].legend(loc = 'upper left')
    #plot various istars with a few different N values
    ax[1, 1].set_title('Figure 4')
    ax[1, 1].plot(xx, t_100, label = 'N = 100')
    ax[1, 1].plot(xx, t_500, label = 'N = 500')
    ax[1, 1].plot(xx, t_1000, label = 'N = 1000')
    ax[1, 1].set_xlabel("istar/(N-1)")
    ax[1, 1].set_ylabel("Wall Time (s)")
    ax[1, 1].legend(loc = "upper left")

    
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
    #Covert strings S and T to base 4
    X = base_4_converter(S)
    Y = base_4_converter(T)
    
    #choose a prime (this prime can be any prime as we assume integer arithmetic takes constant time)
    q = 1871
    bm = (4 ** m ) % q
    #make hashes of all length-m strings in T using heval and Rabin-Karp
    hash_T = heval(Y[: m], 4, q)
    h_dict = {hash_T : [0]}
    for i in range(1, l - m + 1):
        hash_T = ((4 * hash_T - int(Y[i - 1]) * bm + int(Y[i + m - 1])) % q)
        if hash_T in h_dict:
            h_dict[hash_T].append(i)
        else:
            h_dict[hash_T] = [i]

    #Find hash of first m elements in S using heval
    hash_i = heval(X[:m], 4, q)

    #Compare the first m elements of S to all length-m substrings of T, adding to L if matching
    if hash_i in h_dict:
        for i in h_dict[hash_i]:
            if S[:m] == T[i:i + m]:
                L[i].append(0)

    #For the rest of the length-m  strings in S, update hash_i and compare to all hashes in T, adding to L if matching
    for i in range(1, n - m + 1):
        #Update the hash using Rabin-Karp
        hash_i = ((4 * hash_i - int(X[i - 1]) * bm + int(X[i + m - 1])) % q)
        #If the hash is in the dictionary of S length-m strings, then check for a hash collision and append to list.
        if hash_i in h_dict:
            for j in h_dict[hash_i]:
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
    #Convert L to base-10 mod q where Base is original base of L
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
