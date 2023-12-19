
def solution(n):
    if n<=2:
        return 0
    lst_type = {1:{1:1},2:{2:1}}
    for i in range(3,n+1):
        curr_type = {i:1}
        for k in range(1,i):
            curr_type[i-k]=0
            load_type = lst_type[k]
            for j in load_type.keys():
                if j <(i-k):
                    curr_type[i-k]+=load_type[j]
        lst_type[i]=curr_type
    type_ret= 0
    for i in range(1,n):
        type_ret+=lst_type[n][i]
    return type_ret
print(solution(200))