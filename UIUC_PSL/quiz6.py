# 7.
# Question 7
# Question 7- Question 12 are related.
#
# In this problem, you will perform K-means clustering (using Euclidean distance) manually, with K = 2, on a small example with n = 6 observations and p = 2 features.
#
# The observations are as follows.
#
# (1,4), (1,3), (0,4), (5,1), (6,2), (4,0)
#
# Set (1,4) as the centroid for cluster 1 and (1,3) as the centroid for cluster 2. Then
#
# (a) assign each observation to the nearest cluster.
# How many points will be assigned to cluster 1? [n1]
#
# How many points will be assigned to cluster 2? [n2]
#
# (b) update the centroids for the two clsuters.
# What's the x-coordinate of the new centroid for cluster 1? [a1]
#
# What's the x-coordinate of the new centroid for cluster 2? [a2]
#
# Repeat (a) and (b) until convergence. After the algorithm converges,
#
# the x-coordinate of the cluster centroid to which (4,0) belongs is equal to [c1], and
# the size of the cluster to which (4,0) belongs is [c2], i.e., number of points in that cluster including (4,0).

list_points = [(1, 4), (1, 3), (0, 4), (5, 1), (6, 2), (4, 0)]
center_1 = (1, 4)
center_2 = (1, 3)

def get_distance(point, center):
    return ((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2) ** 0.5


def reassign_group(center_1, center_2):
    list_point_center_1 = []
    list_point_center_2 = []
    for point in list_points:
        d1 = get_distance(point, center_1)
        d2 = get_distance(point, center_2)
        
        if d1 > d2:
            print(center_2)
            list_point_center_2.append(point)
        elif d1 == d2:
            raise Exception(f'{center_2} or {center_1}')
        else:
            print(center_1)
            list_point_center_1.append(point)

    return (list_point_center_1, list_point_center_2)

def get_center_point(list_points):
    x = y = 0
    for point in list_points:
        x += point[0]
        y += point[1]
    
    return x/len(list_points), y / len(list_points)

list_point_center_1, list_point_center_2 = reassign_group(center_1, center_2)
center_1 = get_center_point(list_point_center_1)
center_2 = get_center_point(list_point_center_2)

print(center_1, center_2)




# Q13

1,2 (0.3)

3,4 (0.45)


# Q15

0.5 * 0.6 * 0.6

1/2 * 3/5 * 3/5

9 / 50


p(X1|1) * p1 + p(x2|2) * p2 = 1/6 * 1/2 + 1/10 * 1/2 = 1/12 + 1/20 = (5 + 3) / 60 = 2/15


= p(z1=2, x1, x2) / p(x1, x2)


p(z1=2, x1, x2) = p(x1 , x2 | z1=2) * p(z1=2) = 1/2 * 1/10 * (3/5 * 1/10 + 2/5 * 1/6)

p(x1, x2) = p(x1, x2 | z1=1) * p(z1=1) + p(x1, x2 | z1=2) * p(z1=2) = \
    1/2 * 1/10 * (3/5 * 1/10 + 2/5 * 1/6) + \
    1/2 * 1/6 * (3/5 * 1/6 + 2/5 * 1/10)

1/2 * 1/10 * (3/5 * 1/10 + 2/5 * 1/6) / (1/2 * 1/10 * (3/5 * 1/10 + 2/5 * 1/6) + 1/2 * 1/6 * (3/5 * 1/6 + 2/5 * 1/10) )

1/20 * ( 3/50 + 2/30 ) / 1/12 * (3/30 + 2/50)

1/20 * 19/150 / ( 1/12 * 21/150 + 1/20 * 19/150)

1 / (20/12 * 21 /19 + 1)

12 * 19
20 * 21 + 12 * 19

228/6 / (648 / 6

p(z2=2 | x1, x2) = p(x1=1, x2=1 | z2=2) * z2=2 / p(x1, x2)


p(x1=1, x2=1 | z2=2) = p(x1=1, x2=1 | z2=2, z1=1) * z1=2 + p(x1=1, x2=1 | z2=2, z1=2) * z1=2 = 1/6 * 1/10 * 1/2 + 1/10 * 1/10 * 1/2 = 1/120 + 1/200 = (5 + 3) / 600 = 1/75
p(z2=2) = p(z2|z1=1) * z1=1 + p(z2|z1=2) * z1=2 = 2/5 * 1/2 + 3/5 * 1/2 = 1/2


