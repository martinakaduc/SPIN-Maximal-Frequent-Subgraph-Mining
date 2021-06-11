import numpy as np
import sys

def read_graph_corpus(path, label_center_path=None):
    graphs = []
    # label_center = open(label_center_path, 'r', encoding='utf-8')
    label_centers = []
    with open(path, 'r', encoding='utf-8') as file:
        nodes = {}
        edges = {}
        for line in file:
            if 't' in line:
                if len(nodes) > 0:
                    graphs.append((nodes, edges))
                    # if len(graphs) > 9:
                        # break
                nodes = {}
                edges = {}
            if 'v' in line:
                data_line = line.split()
                node_id = int(data_line[1])
                node_label = int(data_line[2])
                nodes[node_id] = node_label
            if 'e' in line:
                data_line = line.split()
                source_id = int(data_line[1])
                target_id = int(data_line[2])
                label = int(data_line[3])
                edges[(source_id, target_id)] = label
        if len(nodes) > 0:
            graphs.append((nodes,edges))
    return graphs

def readGraphs(path):
    rawGraphs = read_graph_corpus(path)
    graphs = []
    num_vertices = []
    num_edges = []

    for graph in rawGraphs:
        numVertices = len(graph[0])
        g = np.zeros((numVertices,numVertices),dtype=int)
        num_vertices.append(0)
        num_edges.append(0)
        for v,l in graph[0].items():
            g[v,v] = l
            num_vertices[-1] += 1
        for e,l in graph[1].items():
            g[e[0],e[1]] = l
            g[e[1],e[0]] = l
            num_edges[-1] += 1
        graphs.append(g)

    print("Average nodes: %.2f" % (sum(num_vertices) / len(graphs)))
    print("Average edges: %.2f" % (sum(num_edges) / len(graphs)))
    return graphs

if __name__ == '__main__':
    readGraphs(sys.argv[1])
