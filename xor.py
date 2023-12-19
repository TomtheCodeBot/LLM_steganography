import math

def solution(start, length):
    ret = start+length*(length-1)
    for i in range(0,length-1):
        for k in range(0,length):
            if length-k>i:
                ret = ret^(start+i*length+k)
            else:
                break
    return ret
def findXOR(n):
    mod = n % 4
    if (mod == 0):
        return n; 
    elif (mod == 1):
        return 1
    elif (mod == 2):
        return n + 1
    elif (mod == 3):
        return 0;    
    
def findXORrange(l, r):
    return (findXOR(l - 1) ^ findXOR(r))

def solution2(start, length):
    ret = None
    test = start
    lst = []
    for i in range(0,length):
        curr_bit = start+i*length
        lst.append(findXORrange(curr_bit,curr_bit+(length-1-i)))
    ret = None
    for i in lst:
        if ret is None:
            ret = i
        else:
            ret= ret^i
    return ret

print(solution(17,4))
print(solution2(17,4))


print(math.log(200000000,2))