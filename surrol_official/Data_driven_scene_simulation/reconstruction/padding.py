import numpy as np
import matplotlib.pyplot as plt


class point:
    def __init__(self,x,y) -> None:
        self.x=x
        self.y=y

class bbox:
    def __init__(self) -> None:
        self.lower=point(999999,99999)
        self.upper=point(0,0)

    def print(self):
        print(self.lower.x,self.lower.y)
        print(self.upper.x,self.upper.y)

def padding(op_map,bottom:float):
    boundary=int(100*bottom)
    box=bbox()
    for i in range(op_map.shape[0]):
        for j in range(op_map.shape[1]):
            if op_map[i][j]==.0:
                continue
            if i<box.lower.x:
                box.lower.x=i
            if i>box.upper.x:
                box.upper.x=i
            if j<box.lower.y:
                box.lower.y=j
            if j>box.upper.y:
                box.upper.y=j
    cnt=0
    pad_points=[]
    for i in range(box.lower.x, box.upper.x+1):
        for j in range(box.lower.y, box.upper.y+1):
            for k in range(boundary):
                if k>op_map[i,j] and op_map[i,j]>0:
                    cnt+=1
                    pad_points.append(np.array([i,j,k],dtype=float)*0.01)

    return cnt, np.array(pad_points)

def visualize_particels(particles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z= 1-particles[:,2] #for better visualization results
    ax.scatter(particles[:,0], particles[:,1],z,c=z)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Particle Positions')
    plt.show()
    plt.savefig('./particles.png')



