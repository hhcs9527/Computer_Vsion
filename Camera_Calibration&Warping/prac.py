import numpy as np 
# library
from shapely.geometry import Point, Polygon

# Assign specific index(boundary) to 0
A = np.array([[2,2,2,2,2],[2,2,2,2,2],[2,2,2,2,2],[2,2,2,2,2]])


print(np.sum(A, axis =0))



#poly = Polygon(bounadry)
#r,c = zip(*bounadry)
#minx = min(r)
#maxx = max(r)
#miny = min(c)
#maxy = max(c)
#A[r,c] = 0
#print(minx, miny, maxx, maxy)


#for i in coor:
##    for j in range(5):
#    m = np.array(i)
#    T = poly_determine(m, bounadry)
#    if T:
#        A[i[0],i[1]] = 0
#        #print('({},{}) is {} with value {}'.format(i,j, T, v))


#print(A)
# fill in the boundary



#A[pos] = 8

#print(Q)
