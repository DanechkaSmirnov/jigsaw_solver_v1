import glob
import numpy as np
from math import sqrt
from numpy import dot
from numpy.linalg import norm


def open_file(path):
    file = open(path).read().split()
    if file[1] == '#':
        for i in range(5):
            file.pop(1)
    file = tuple(map(int, file[1:]))
    size = list(file[0:2])
    size.append(3)
    img = np.reshape(file[3:], size)
    return img.tolist()

def img_in_folder(path):
    return glob.glob(f"{path}*.ppm")

def get_corns(img):
    up = img[0]
    right = [row[-1] for row in img]
    down = img[-1][::-1]
    left = [row[0] for row in img[::-1]]
    return [up, right, down, left]

def rgb_distance(corn1, corn2):
    err = 0
    corn2 = corn2[::-1]
    for i in range(len(corn1)):
        err+=(abs(corn1[i][0] - corn2[i][0])+abs(corn1[i][1] - corn2[i][1])+abs(corn1[i][2] - corn2[i][2]))
    return err


def find_min(matrix):
    mymin = min([min(r) for r in matrix])
    for i in range(len(matrix)):
        if mymin in matrix[i]:
            return i, matrix[i].index(mymin), mymin
            break
            
            
def get_index_to_inf(a, b, x):
    indexes = []
    a = a//4
    b = b//4
    list1 = [i for i in range(a*4, (a+1)*4)]
    list2 = [i for i in range(b*4, (b+1)*4)] 
    for i in list1:
        for j in list2:
            indexes.append([i, j])
            indexes.append([j, i])
    for i in range(x):
        indexes.append([i, a])
        indexes.append([a, i])
        indexes.append([i, b])
        indexes.append([b, i])
    return indexes

def count_of_puzzles(matr):
    count = 0
    for stripe in matr:
        count+= len(stripe)-stripe.count(None)
    print(count)
    
def get_index_in_jigsaw(matr, num):
    for i in range(len(matr)):
        if num in matr[i]:
            return i, matr[i].index(num)
        
def get_rotation(p1, p2):
    #p1 - puzzle, p2 - side
    if p1[0] == p2[0]:
        if p1[1] + 1 == p2[1]:
            rotation = 3
        elif p1[1] - 1 == p2[1]:
            rotation = 1
    elif p1[1] == p2[1]:
        if p1[0] + 1 == p2[0]:
            rotation = 0
        elif p1[0]-1==p2[0]:
            rotation = 2
    return rotation
        
    
def get_string(jig):
    res = []
    for string in jig:
        res.append([i for i in string if isinstance(i, str)])
    answ = [e for e in res if e]
    return answ

def rotateMatrix(mat):
    return list(zip(*mat))[::-1]

def first_angle(mat):
    for i in range(len(mat)):
        if None in mat[i]:
            return [i, mat[i].index(None)]
        
def solve_puzzle(folder):
    folder = '/Users/danny/Downloads/data/0000_0003_0002/tiles/'
    file_names = sorted(img_in_folder(folder))
    imgs = []
    for path in file_names:
        imgs.append(open_file(path))
    corns = []
    for img in imgs:
        corns.extend(get_corns(img))


    from math import inf
    adj_matrix = [[None for _ in range(len(corns))] for _ in range(len(corns))]
    inf_matrix = [[inf for _ in range(len(corns))] for _ in range(len(corns))]
    for i in range(len(corns)):
        for j in range(i, len(corns)):
            if i//4 == j//4:
                adj_matrix[i][j] = inf
                adj_matrix[j][i] = inf
            else:

                adj_matrix[i][j] = round(rgb_distance(corns[i], corns[j]))
                adj_matrix[j][i] = adj_matrix[i][j]
    cons = []
    for i in range(len(adj_matrix)):
        cons.append([i, adj_matrix[i].index(min(adj_matrix[i])), min(adj_matrix[i])])
        adj_matrix[adj_matrix[i].index(min(adj_matrix[i]))][i] = inf

    cons.sort(key=lambda x: x[2])

    num_of_puzzles = len(cons)//4
    jigsaw = [[None for _ in range(num_of_puzzles*2+1)] for _ in range(num_of_puzzles*2+1)]
    jigsaw[num_of_puzzles][num_of_puzzles] = f'{cons[0][0]//4}'
    jigsaw[num_of_puzzles-1][num_of_puzzles] = cons[0][0]//4*4
    jigsaw[num_of_puzzles][num_of_puzzles+1] = cons[0][0]//4*4+1
    jigsaw[num_of_puzzles+1][num_of_puzzles] = cons[0][0]//4*4+2
    jigsaw[num_of_puzzles][num_of_puzzles-1] = cons[0][0]//4*4+3
    solved = 1
    rotations = {cons[0][0]//4:0}
    cord = [num_of_puzzles, num_of_puzzles]
    inserted = [cons[0][0]//4]
    inserted_index = [cord]
    tofind = {cons[0][0]//4*4:cord, cons[0][0]//4*4+1:cord, cons[0][0]//4*4+2:cord, cons[0][0]//4*4+3:cord}
    while solved != num_of_puzzles:
        father = 0
        mother = 0
        for line in cons:
            if line[0]//4 in inserted:
                father = line[1]
                mother = line[0]
                cons.remove(line)
                break
            elif line[1]//4 in inserted:
                father = line[0]
                mother = line[1]
                cons.remove(line)
                break
        if father//4 in inserted:
            continue
        fath_index = get_index_in_jigsaw(jigsaw, mother)
        moth_index = tofind[mother]
        if fath_index == None or moth_index == None:
            continue
        moth_dir = get_rotation(fath_index, tofind[mother])
        fath_dir = ((moth_dir+2)%4+4-father%4)%4
        jigsaw[fath_index[0]][fath_index[1]] = f'{father//4}'
        inserted_index.append(list(fath_index))
        rotations[father//4] = fath_dir
        if [fath_index[0]-1, fath_index[1]] not in inserted_index:
            jigsaw[fath_index[0]-1][fath_index[1]] = father//4*4+(-fath_dir)%4
        if [fath_index[0], fath_index[1]+1] not in inserted_index:
            jigsaw[fath_index[0]][fath_index[1]+1] = father//4*4+(-fath_dir+1)%4
        if [fath_index[0]+1, fath_index[1]] not in inserted_index:
            jigsaw[fath_index[0]+1][fath_index[1]] = father//4*4+(-fath_dir+2)%4
        if [fath_index[0], fath_index[1]-1] not in inserted_index:
            jigsaw[fath_index[0]][fath_index[1]-1] = father//4*4+(-fath_dir+3)%4
        solved+=1
        inserted.append(father//4)
        tofind[father//4*4] = fath_index
        tofind[father//4*4+1] = fath_index
        tofind[father//4*4+2] = fath_index
        tofind[father//4*4+3] = fath_index

    locations = get_string(jigsaw)
    locations = [item for sublist in locations for item in sublist]


    for rotor in rotations.keys():
        if rotations[rotor] != 0:
            for _ in range(4-rotations[rotor]):
                imgs[rotor] = rotateMatrix(imgs[rotor])
    if len(jigsaw) < len(jigsaw[0]):
        jigsaw = rotateMatrix(jigsaw)

    final_image = [[None for _ in range(1200)] for _ in range(900)]
    for block in locations:
        img = imgs[int(block)]
        pos = first_angle(final_image)
        for i in range(len(img)):
            for j in range(len(img[0])):
                final_image[pos[0]+i][pos[1]+j] = img[i][j]


    image = 'P3\n 1200 900\n 250\n'
    for line in final_image:
        for pixel in line:
            image+=f'{pixel[0]} {pixel[1]} {pixel[2]} '
        image+='\n'
    return image

if __name__ == "__main__":
    folder = sys.argv[1]
    solve_puzzle(folder)
    

