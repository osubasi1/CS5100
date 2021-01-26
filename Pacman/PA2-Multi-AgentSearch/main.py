# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import time

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')


def optLcs(X, Y, Z):
    start_time = time.time()
    l1 = len(X)
    l2 = len(Y)
    l3 = len(Z)

    opt_length = [[[0 for i in range(l3 + 1)] for j in range(l2 + 1)] for k in range(l1 + 1)]


    for i in range(l1 + 1):
        for j in range(l2 + 1):
            for k in range(l3 + 1):
                if (i == 0 or j == 0 or k == 0):
                    opt_length[i][j][k] = 0

                elif X[i - 1] == Y[j - 1] and X[i - 1] == Z[k - 1]:
                    opt_length[i][j][k] = opt_length[i - 1][j - 1][k - 1] + 1

                else:
                    opt_length[i][j][k] = max(max(opt_length[i - 1][j][k], opt_length[i][j - 1][k]), opt_length[i][j][k - 1])





    lcs =""

    i = l1
    j = l2
    k = l3

    while i > 0 and j > 0 and k > 0:

        # If current character is same for all three then it is in lcs
        if X[i - 1] == Y[j - 1] and X[i - 1] == Z[k - 1]:
            #lcs[index - 1] = X[i - 1]
            lcs = X[i-1] + lcs
            i -= 1
            j -= 1
            k -= 1


        elif opt_length[i - 1][j][k] > opt_length[i][j - 1][k - 1] and opt_length[i - 1][j][k] > opt_length[i][j][k - 1]:
            i -= 1
        elif opt_length[i][j - 1][k] > opt_length[i - 1][j][k - 1] and opt_length[i][j - 1][k] > opt_length[i][j][k - 1]:
            j -= 1
        else:
            k -= 1
    print("My program took", time.time() - start_time, "to run")

    return "".join(lcs)



X = "uwytubiweutpowegbiutpoweeniwotiweqz"
Y = "fdsjkhfbklsdfigkjfdbensdjdlqa"
Z = "cmnvbcxmbimxcnbvmgxmnbvcxbexmbnnaz"
A ="If you want to include text as is like the one you are just reading all you need to is use the PRE tag. This will allow you to displaytext exactly as you type it.  It will give you a line break whereveryou press Enter, like here...."
B ="If you want to include text as is like the one you are just reading all you need to is use the PRE tag. This will allow you to displaytext exactly as you type it.  It will give you a line break whereveryou press Enter, like here...."
C ="If you want to include text as is like the one you are just reading all you need to is use the PRE tag. This will allow you to displaytext exactly as you type it.  It will give you a line break whereveryou press Enter, like here...."

#print(optLcs(A, B, C))
INT_MIN = -32767






def cutRod(D, X):
    opt = [0 for x in range(len(X) + 1)]
    opt[0] = 0
    choice = [0 for x in range(len(X) + 1)]


    # Build the table val[] in bottom up manner and return
    # the last entry from the table
    for i in range(1, len(X)+1):
        max_val = float("-inf")
        for j in range(i):
            if i < len(X):
                if D[i] > X[i]:
                    D[i] = X[i]
            if max_val < D[j] + opt[i - j - 1]:
                max_val = D[j] + opt[i - j - 1]
                choice[i] = j+1
        opt[i] = max_val


    n = len(choice)-1
    deneme =[]
    while n > 0:
        deneme.append(choice[n])
        n = n - choice[n]
    deneme.reverse()
    for i in range(len(deneme)-1):
        deneme[i+1]= deneme[i]+deneme[i+1]


    return opt,choice


# Driver program to test above functions
arr = [1, 10, 10, 50]
arr0=[10,20,4,80]
arr2 = [1,5,8,9,10,17,17,20,24,30]
arr3= [4,2,3,1]
size = len(arr)
xx=[2,10]
yy =[[2,10],[1,5],[3,6],[3,9],[3,4],[4,5],[4,9],[7,8],[9,10]]
nn=7
def deneme(y,n):

    S = set()
    prev = y[1]
    addVal =1
    S.add(addVal)

    while prev[1] < y[0][1]:
        d = 0
        for i in range(addVal,len(y)):
            if y[i][0] <= prev[1]  and d < y[i][1] - prev[1]:
                d = y[i][1] - prev[1]
                print("prev is:", prev)
                addVal = i
        prev = y[addVal]
        S.add(addVal)





    return S
print(deneme(yy,nn))





