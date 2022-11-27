p=[[0.111 ,
    0.1223,
    224.09]]

p = p.flatten()


p_p = sum(f) / len(f) * 100

score = round(p_p)
positive = 'positive' + str(score)

print(positive)