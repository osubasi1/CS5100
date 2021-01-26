import queue
x = [[8, 10], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [2, 3], [2, 4], [4, 6], [5, 6], [2, 5]]

y = [[12, 30],[1,2],[1, 3],[1,4], [1, 5], [1, 6],[2,3], [2, 4], [2, 5], [2, 6], [3, 4],
     [3, 5], [3, 6], [4, 5], [4, 6], [5, 6], [7, 8], [7, 9], [7, 10], [7, 11], [7, 12], [8, 9],
     [8, 10], [8, 11], [8, 12], [9, 10], [9, 11], [9, 12], [10, 11], [10, 12],[11,12]
     ]

# def party(n):
#
#
#     S = set(range(1,n[0][0]+1))
#     pairDict = {}
#     for pair in n[1:]:
#         if pair[0] in pairDict.keys():
#             pairDict[pair[0]].append(pair[1])
#         else:
#             pairDict[pair[0]] = [pair[1]]
#         if pair[1] in pairDict.keys():
#             pairDict[pair[1]].append(pair[0])
#         else:
#             pairDict[pair[1]] = [pair[0]]
#
#     #print(pairDict)
#     for k in pairDict:
#         if len(pairDict[k]) < 5:
#             for i in pairDict:
#                 if k in pairDict[i]:
#                     pairDict[i].remove(k)
#
#
#
#
#
#     return S
# party(x)
# party(y)
#
# print(len(party(x)))
# print(*party(x))

#print(len(party(y)))
#print(*party(y))

def party2(n):

    S = set(range(1,n[0][0]+1))
    pairDict = {}
    for i in (range(1,n[0][0]+1)):
        pairDict[i] = set()


    for each in n[1:]:
        pairDict[each[0]].add(each[1])
        pairDict[each[1]].add(each[0])


    for person in pairDict:
        if len(S) - len(pairDict[person]) - 1 < 5:
            if person in S:
                S.remove(person)

    for k in pairDict:
        if len(pairDict[k]) < 5 or len(S) - len(pairDict[k]) - 1 < 5:
            if k in S :
                S.remove(k)
            for i in pairDict:
                if k in pairDict[i]:
                    pairDict[i].remove(k)
                    if len(pairDict[i]) < 5 or len(S) - len(pairDict[i]) - 1 < 5:
                        if i in S:
                            S.remove(i)


    return (S)





    # print(S)
    # print (pairDict)
# print(party2(y))
# print(party2(x))


def party3(n):

    S = set(range(1,n[0][0]+1))
    pairDict = {}
    for i in (range(1,n[0][0]+1)):
        pairDict[i] = set()


    for each in n[1:]:
        pairDict[each[0]].add(each[1])
        pairDict[each[1]].add(each[0])

    invite = [True]*(n[0][0]+1)

    Q = queue.Queue()
    for person in pairDict:
        invite[person] = True
        if len(pairDict[person]) < 5 or len(S) - len(pairDict[person]) -1 < 5:
            Q.put(person)
            invite[person] = False


    while not Q.empty():

        person = Q.get()
        S.remove(person)
        for each in pairDict:
            if person in pairDict[each]:
                pairDict[each].remove(person)
            if (len(pairDict[each]) < 5 or len(S) - len(pairDict[each]) -1 < 5) and invite[each] == True:
                Q.put(each)
                invite[each] = False


    print(S)

party3(y)

