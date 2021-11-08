import numpy as np
import glob
import sys
from math import inf
import os

filler = np.zeros((900, 1200, 3)).astype(int)

def read_folder(folder):
    try:
        file_names = sorted(glob.glob(f"{folder}*.ppm"))
        imgs = []
        for path in file_names:
            file = open(path)
            line = file.readline()
            while len(line.split()) != 2:
                line = file.readline()
            shape = list(map(int, line.split()))
            file.readline()
            imgs.append(np.fromstring(file.read(), dtype=int, sep=' ').reshape((shape[0], shape[1], 3)))
        return imgs
    except Exception as err:
        image = 'P3\n 1200 900\n 255\n'
        for line in filler.astype(int):
            for pixel in line:
                image += f'{pixel[0]} {pixel[1]} {pixel[2]} '
            image += '\n'

        with open(os.getcwd()+'/image.ppm', 'w') as file:
            file.write(image)


def get_image(jig):
    try:
        res = []
        for string in jig:
            res.append([i for i in string if isinstance(i, int)])
        answ = [e for e in res if e]
        return answ
    except Exception as err:
        image = 'P3\n 1200 900\n 255\n'
        for line in filler.astype(int):
            for pixel in line:
                image += f'{pixel[0]} {pixel[1]} {pixel[2]} '
            image += '\n'

        with open(os.getcwd()+'/image.ppm', 'w') as file:
            file.write(image)



def get_edges(img):
    try:
        img_len = len(img)
        up = img[0:1, :, :].reshape((1, img_len, 3))
        right = img[:, -1:-2:-1, :].reshape((1, img_len, 3))
        down = img[-1:-2:-1, ::-1, :].reshape((1, img_len, 3))
        left = img[::-1, 0:1, :].reshape((1, img_len, 3))
        return [up, right, down, left]
    except Exception as err:
        image = 'P3\n 1200 900\n 255\n'
        for line in filler.astype(int):
            for pixel in line:
                image += f'{pixel[0]} {pixel[1]} {pixel[2]} '
            image += '\n'

        with open(os.getcwd()+'/image.ppm', 'w') as file:
            file.write(image)



class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def apply_union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def kruskal_algo(self):
        try:
            result = []
            i, e = 0, 0
            self.graph = sorted(self.graph, key=lambda item: item[2])
            parent = []
            rank = []
            for node in range(self.V):
                parent.append(node)
                rank.append(0)
            while e < self.V - 1:
                u, v, w = self.graph[i]
                i = i + 1
                x = self.find(parent, u)
                y = self.find(parent, v)
                if x != y:
                    e = e + 1
                    result.append([u, v, w])
                    self.apply_union(parent, rank, x, y)
            res = []
            for u, v, length in result:
                res.append([u, v, length])
            return res
        except Exception as err:
            image = 'P3\n 1200 900\n 255\n'
            for line in filler.astype(int):
                for pixel in line:
                    image += f'{pixel[0]} {pixel[1]} {pixel[2]} '
                image += '\n'

            with open(os.getcwd()+'/image.ppm', 'w') as file:
                file.write(image)



def paired_cons(cons):
    cleaned = []
    cons_raw = [[x[0], x[1]] for x in cons]
    pairs = [x for x in cons if [x[1], x[0]] in cons_raw]
    return pairs


def get_index_of_tile(jigsaw, tile):
    try:
        for i in range(len(jigsaw)):
            if tile in jigsaw[i]:
                return [i, jigsaw[i].index(tile)]
    except Exception as err:
        image = 'P3\n 1200 900\n 255\n'
        for line in filler.astype(int):
            for pixel in line:
                image += f'{pixel[0]} {pixel[1]} {pixel[2]} '
            image += '\n'

        with open(os.getcwd()+'/image.ppm', 'w') as file:
            file.write(image)

            
            
def solve(folder):
    try:
        imgs = read_folder(folder)
        num_of_tiles = len(imgs)
        corns = []
        for img in imgs:
            corns.extend(get_edges(img))

        adj_matrix = [[None for _ in range(num_of_tiles * 4)] for _ in range(num_of_tiles * 4)]
        for i in range(num_of_tiles * 4):
            for j in range(num_of_tiles * 4):
                if i // 4 != j // 4:
                    adj_matrix[i][j] = abs((corns[i][:, ::-1, :].astype(np.int8)) - corns[j].astype(np.int8)).sum()
                    adj_matrix[j][i] = adj_matrix[i][j]
                else:
                    adj_matrix[i][j] = inf
                    adj_matrix[j][i] = inf

        cons = []
        for i in range(num_of_tiles * 4):
            cons.append([i, adj_matrix[i].index(min(adj_matrix[i])), min(adj_matrix[i])])

        cons.sort(key=lambda x: x[2])
        pairs = paired_cons(cons)
        tiles = [[x[0] // 4, x[1] // 4, x[2]] for x in pairs]
        graph = Graph(len(imgs))
        for v in tiles:
            graph.add_edge(*v)
        nodes = graph.kruskal_algo()
        cons_clean = [[x[0], x[1]] for x in cons if [x[0] // 4, x[1] // 4, x[2]] in nodes]
        cons = cons_clean

        inserted = [cons[0][0] // 4]
        rotated = {cons[0][0] // 4: 0}
        indexes = {cons[0][0] // 4: [num_of_tiles // 2, num_of_tiles // 2]}
        add = []
        while len(cons) > 0:
            for con in cons:
                if con[0] // 4 in inserted:
                    mother = con[0]
                    father = con[1]
                    cons.remove(con)
                    break
                elif con[1] // 4 in inserted:
                    mother = con[1]
                    father = con[0]
                    cons.remove(con)
                    break
            moth_dir = (mother % 4 - rotated[mother // 4]) % 4  ## 3 - 0 = 3
            moth_index = indexes[mother // 4]
            fath_rot = ((father % 4 - (moth_dir + 2) % 4)) % 4  ##3 - 1 = 2
            rotated[father // 4] = fath_rot
            inserted.append(father // 4)
            if moth_dir == 0:
                indexes[father // 4] = [moth_index[0] - 1, moth_index[1]]
            if moth_dir == 1:
                indexes[father // 4] = [moth_index[0], moth_index[1] + 1]
            if moth_dir == 2:
                indexes[father // 4] = [moth_index[0] + 1, moth_index[1]]
            if moth_dir == 3:
                indexes[father // 4] = [moth_index[0], moth_index[1] - 1]

        jigsaw = [[None for _ in range(num_of_tiles * 2 + 1)] for _ in range(num_of_tiles * 2 + 1)]
        for tile in range(num_of_tiles):
            index = indexes[tile]
            jigsaw[index[0]][index[1]] = tile
        locations = get_image(jigsaw)

        if len(np.array(locations, dtype=object).shape) == 2:
            flag = 0
            if len(locations) > len(locations[0]):
                locations = np.rot90(np.array(locations))
                flag = 1

        final_image = np.ndarray((0, 1200, 3))

        for i in rotated.keys():
            for _ in range((rotated[i] + flag)):
                imgs[i] = np.rot90(imgs[i])

        for line in locations:
            strip = np.ndarray((len(imgs[0]), 0, 3))
            for block in line:
                strip = np.append(strip, imgs[int(block)][:, :, :], axis=1)

            final_image = np.append(final_image, strip, axis=0)

        image = 'P3\n 1200 900\n 255\n'
        for line in final_image.astype(int):
            for pixel in line:
                image += f'{pixel[0]} {pixel[1]} {pixel[2]} '
            image += '\n'

        with open(os.getcwd()+'/image.ppm', 'w') as file:
            file.write(image)
    except Exception as err:
        image = 'P3\n 1200 900\n 255\n'
        for line in filler.astype(int):
            for pixel in line:
                image += f'{pixel[0]} {pixel[1]} {pixel[2]} '
            image += '\n'

        with open(os.getcwd()+'/image.ppm', 'w') as file:
            file.write(image)


if __name__ == "__main__":
    folder = sys.argv[1]+'/'
    solve(folder)
