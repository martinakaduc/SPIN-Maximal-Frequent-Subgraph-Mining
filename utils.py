import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt

def write_graph(graphs, file_name):
    with open(file_name, mode='w', encoding='utf-8') as file:
        for i, graph in enumerate(graphs):
            visited = [False]*len(graph)
            labelNodes = graph.diagonal()

            file. writelines("t # %d\n" % i)
            for node_i, node_label in enumerate(labelNodes):
                file.write("v %d %d\n" % (node_i, node_label))

            startNode = np.argmax(labelNodes)
            queue = []
            queue.append(startNode)

            while queue:
                s = queue.pop(0)
                visited[s] = True

                edge_list = np.where(graph[s]>0)[0].tolist()
                # Sort edge
                node_list = [graph[x, x] for x in edge_list]

                edge_list = list(sorted(zip(edge_list, node_list), key=lambda x: x[1], reverse=True))
                # print(edge_list)

                for i, _ in edge_list:
                    if not visited[i]:
                        if i not in queue:
                            queue.append(i)
                        file.writelines("e %d %d %d\n" % (s, i, graph[s,i]))

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
    return graphs#[10:]

def readGraphs(path):
    rawGraphs = read_graph_corpus(path)
    graphs = []
    for graph in rawGraphs:
        numVertices = len(graph[0])
        g = np.zeros((numVertices,numVertices),dtype=int)
        for v,l in graph[0].items():
            g[v,v] = l
        for e,l in graph[1].items():
            g[e[0],e[1]] = l
            g[e[1],e[0]] = l
        graphs.append(g)
    return graphs

def plotGraph(graph : np.ndarray,isShowedID=True):
    edges = []
    edgeLabels = {}
    for i in range(graph.shape[0]):
        indices = np.where(graph[i][i+1:] > 0)[0]
        for id in indices:
            edges.append([i,i+id+1])
            edgeLabels[(i,i+id+1)] = graph[i,i+id+1]
    # print(edges,edgeLabels)
    # exit(0)
    G = nx.Graph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    plt.figure()
    nodeLabels = {node:node for node in G.nodes()} if isShowedID else {node:graph[node,node] for node in G.nodes()}
    nx.draw(G,pos,edge_color='black',width=1,linewidths=1,
        node_size=500,node_color='pink',alpha=0.9,
        labels=nodeLabels)

    nx.draw_networkx_edge_labels(G,pos,edge_labels=edgeLabels,font_color='red')
    plt.axis('off')
    # plt.savefig('./figures/{}.png'.format(np.array2string(graph[0])),format='PNG')
    plt.show()
