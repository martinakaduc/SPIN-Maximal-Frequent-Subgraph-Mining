import numpy as np

def encodeGraph(graph):
    visited = [False]*len(graph)
    labelNodes = graph.diagonal()
    if len(labelNodes) == 0: return "$#"
    startNode = np.argmax(labelNodes)

    queue = []
    queue.append(startNode)
    code = str(graph[startNode,startNode]) + '$'

    while queue:
        s = queue.pop(0)
        visited[s] = True
        levelStr = ''

        edge_list = np.where(graph[s]>0)[0].tolist()
        # Sort edge
        node_list = [graph[x, x] for x in edge_list]

        edge_list = list(sorted(zip(edge_list, node_list), key=lambda x: x[1], reverse=True))
        # print(edge_list)

        for i, _ in edge_list:
            if not visited[i]:
                if i not in queue:
                    queue.append(i)
                levelStr += str(graph[s,i]) + "_" + str(graph[i,i]) + "_"
                visited[i] = True

        if levelStr != '':
            code += levelStr[:-1] +  '$'

    code += '#'

    return code

def canonicalForm(graph: np.ndarray, embeddings=None):
    return {"tree": graph, "code": encodeGraph(graph)}
