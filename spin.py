from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import copy
import itertools

import numpy as np
from typing import List
from algorithm import encodeGraph

from graph import Graph
from graph import AUTO_EDGE_ID
from graph import Graph
from graph import VACANT_GRAPH_ID
from graph import VACANT_VERTEX_LABEL

class S_structure():
    def __init__(self, input_data={}):
        self.data = input_data

    def add_graph(self, canonical, tree, freq):
        self.data[canonical] = {"tree": tree, "freq": freq}

    def __iter__(self):
        return iter(self.data.items())

    def append(self, key, value):
        if key not in self.data.keys():
            self.data[key] = value

    def get_length(self):
        return len(self.data)

def sub_S_structure(here, other):
    new_dict = dict()
    result = S_structure(new_dict)
    other_keys = list(other.data.keys())

    for key, value in here.data.items():
        if key not in other_keys:
            result.append(key, value)

    return result

class SPIN():
    def __init__(self,
                database_file_name,
                min_support=10,
                min_num_vertices=1,
                max_num_vertices=float('inf'),
                max_ngraphs=float('inf'),
                is_undirected=True,
                verbose=False):

        self._database_file_name = database_file_name
        self.graphs = dict()
        self._max_ngraphs = max_ngraphs
        self._is_undirected = is_undirected
        self._min_support = min_support
        self._min_num_vertices = min_num_vertices
        self._max_num_vertices = max_num_vertices
        self._support = 0
        self._frequent_size1_subgraphs = dict()
        self._frequent_edges = dict()
        self._frequent_subgraphs = dict()
        self._counter = itertools.count()
        self._verbose = verbose
        self._loop_count = 0

        if self._max_num_vertices < self._min_num_vertices:
            print('Max number of vertices can not be smaller than '
                  'min number of that.\n'
                  'Set max_num_vertices = min_num_vertices.')
            self._max_num_vertices = self._min_num_vertices

        self._read_graphs()

    def _read_graphs(self):
        self.graphs = dict()
        with codecs.open(self._database_file_name, 'r', 'utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
            tgraph, graph_cnt = None, 0
            for i, line in enumerate(lines):
                cols = line.split(' ')
                if cols[0] == 't':
                    if tgraph is not None:
                        self.graphs[graph_cnt] = tgraph
                        graph_cnt += 1
                        tgraph = None
                    if cols[-1] == '-1' or graph_cnt >= self._max_ngraphs:
                        break
                    tgraph = Graph(graph_cnt,
                                   is_undirected=self._is_undirected,
                                   eid_auto_increment=True)
                elif cols[0] == 'v':
                    tgraph.add_vertex(cols[1], cols[2])
                elif cols[0] == 'e':
                    tgraph.add_edge(AUTO_EDGE_ID, cols[1], cols[2], cols[3])
            # adapt to input files that do not end with 't # -1'
            if tgraph is not None:
                self.graphs[graph_cnt] = tgraph
        return self

    def _report_size1(self, g, support):
        g.display()
        print('\nSupport: {}'.format(support))
        print('\n-----------------\n')

    def _build_1edge_tree(self):
        vevlb_counter = collections.Counter()
        vevlb_counted = set()
        vevlb_dict = dict()

        for g in self.graphs.values():
            for v in g.vertices.values():
                for to, e in v.edges.items():
                    vlb1, vlb2 = v.vlb, g.vertices[to].vlb
                    vid1, vid2 = v.vid, g.vertices[to].vid

                    if self._is_undirected and vlb1 < vlb2:
                        vlb1, vlb2 = vlb2, vlb1
                        vid1, vid2 = vid2, vid1

                    if (g.gid, (vlb1, e.elb, vlb2)) not in vevlb_counted:
                        vevlb_counter[(vlb1, e.elb, vlb2)] += 1
                    vevlb_counted.add((g.gid, (vlb1, e.elb, vlb2)))

                    if (vlb1, e.elb, vlb2) not in vevlb_dict:
                        vevlb_dict[(vlb1, e.elb, vlb2)] = {}

                    if g.gid not in vevlb_dict[(vlb1, e.elb, vlb2)]:
                        vevlb_dict[(vlb1, e.elb, vlb2)][g.gid] = []

                    if [vid1, vid2] not in vevlb_dict[(vlb1, e.elb, vlb2)][g.gid] and [vid2, vid1] not in vevlb_dict[(vlb1, e.elb, vlb2)][g.gid]:
                        vevlb_dict[(vlb1, e.elb, vlb2)][g.gid].append([vid1, vid2])

        self._frequent_edges = vevlb_dict

        for vevlb, cnt in vevlb_counter.items():
            if cnt >= self._min_support:
                g = Graph(gid=next(self._counter),
                                  is_undirected=self._is_undirected)
                g.add_vertex(0, vevlb[0])
                g.add_vertex(1, vevlb[2])
                g.add_edge(AUTO_EDGE_ID, 0, 1, vevlb[1])

                # Mark edge as frequent
                for g_key in self.graphs:
                    if self.graphs[g_key].gid in vevlb_dict[vevlb]:
                        for pair in vevlb_dict[vevlb][self.graphs[g_key].gid]:
                            self.graphs[g_key].set_freq_edge(pair[0], pair[1])

                # Append into set
                g_cannonical = g.get_cannonical_tree()

                self._frequent_size1_subgraphs[g_cannonical] = {"tree": g, "freq": vevlb_dict[vevlb]}

                # if self._min_num_vertices <= 1:
                #     self._report_size1(g, support=cnt)

        # print(self._frequent_size1_subgraphs)
        if self._min_num_vertices > 1:
            self._counter = itertools.count()

        return copy.deepcopy(self._frequent_size1_subgraphs)

    def _expand_1node(self, graph, encodeX):
        new_dict = dict()
        result = S_structure(new_dict)
        print("========================================================")
        print("Expand: ", encodeX)
        # Expand & recalculate which graph hold
        # for vid, vertice in graph["tree"].get_leaf_vertice().items():
        for vid, vertice in graph["tree"].vertices.items():
            freq_edge_set = set()

            for gid in graph["freq"]:
                # for vv in graph["freq"][gid]:
                #     vid_in_graph = vv[vid]
                vid_in_graph = graph["freq"][gid][0][vid]
                freq_edge_set = freq_edge_set | self.graphs[gid].get_freq_edges(vid_in_graph)

            list_to_edge = [graph["tree"].vertices[x].vlb for x in vertice.edges]

            for edge in freq_edge_set:
                if vertice.vlb == edge[0] and edge[2] not in list_to_edge:
                    intersect_graph = list(set(graph["freq"].keys()) & set(self._frequent_edges[edge].keys()))
                    if len(intersect_graph) < self._min_support:
                        continue

                    real_appearance = {}
                    for gid in intersect_graph:
                        real_appearance[gid] = [graph["freq"][gid][0] + [self._frequent_edges[edge][gid][0][1]]]

                    exp_g = copy.deepcopy(graph["tree"])
                    new_vid = exp_g.get_num_vertices()
                    exp_g.add_vertex(new_vid, edge[2])
                    exp_g.add_edge(AUTO_EDGE_ID, vid, new_vid, edge[1])
                    exp_canonical = exp_g.get_cannonical_tree()
                    if exp_canonical > encodeX:
                        result.add_graph(exp_canonical, exp_g, real_appearance)


                elif vertice.vlb == edge[2] and edge[0] not in list_to_edge:
                    intersect_graph = list(set(graph["freq"].keys()) & set(self._frequent_edges[edge].keys()))
                    if len(intersect_graph) < self._min_support:
                        continue

                    real_appearance = {}
                    for gid in intersect_graph:
                        real_appearance[gid] = [graph["freq"][gid][0] + [self._frequent_edges[edge][gid][0][0]]]

                    exp_g = copy.deepcopy(graph["tree"])
                    new_vid = exp_g.get_num_vertices()
                    exp_g.add_vertex(new_vid, edge[0])
                    exp_g.add_edge(AUTO_EDGE_ID, vid, new_vid, edge[1])

                    exp_canonical = exp_g.get_cannonical_tree()
                    if exp_canonical > encodeX:
                        result.add_graph(exp_canonical, exp_g, real_appearance)

            # for edge in self._frequent_edges:
            #     list_to_edge = [graph["tree"].vertices[x].vlb for x in vertice.edges]
            #     if vertice.vlb == edge[0] and edge[2] not in list_to_edge:
            #         intersect_graph = list(set(graph["freq"].keys()) & set(self._frequent_edges[edge].keys()))
            #         if len(intersect_graph) < self._min_support:
            #             continue
            #
            #         real_appearance = {}
            #         for gid in intersect_graph:
            #             if graph["freq"][gid][0][vid] == self._frequent_edges[edge][gid][0][0]:
            #                 real_appearance[gid] = [graph["freq"][gid][0] + [self._frequent_edges[edge][gid][0][1]]]
            #
            #         if len(real_appearance) >= self._min_support:
            #             exp_g = copy.deepcopy(graph["tree"])
            #             new_vid = exp_g.get_num_vertices()
            #             exp_g.add_vertex(new_vid, edge[2])
            #             exp_g.add_edge(AUTO_EDGE_ID, vid, new_vid, edge[1])
            #             # print(exp_g.get_cannonical_tree())
            #             result.add_graph(exp_g.get_cannonical_tree(), exp_g, real_appearance)
            #
            #     elif vertice.vlb == edge[2] and edge[0] not in list_to_edge:
            #         intersect_graph = list(set(graph["freq"].keys()) & set(self._frequent_edges[edge].keys()))
            #         if len(intersect_graph) < self._min_support:
            #             continue
            #
            #         real_appearance = {}
            #         for gid in intersect_graph:
            #             if graph["freq"][gid][0][vid] == self._frequent_edges[edge][gid][0][1]:
            #                 real_appearance[gid] = [graph["freq"][gid][0] + [self._frequent_edges[edge][gid][0][0]]]
            #
            #         if len(real_appearance) >= self._min_support:
            #             exp_g = copy.deepcopy(graph["tree"])
            #             new_vid = exp_g.get_num_vertices()
            #             exp_g.add_vertex(new_vid, edge[0])
            #             exp_g.add_edge(AUTO_EDGE_ID, vid, new_vid, edge[1])
            #             result.add_graph(exp_g.get_cannonical_tree(), exp_g, real_appearance)
        print(result.data.keys())
        return result

    def _generic_tree_explorer(self, C, R):
        new_dict = dict()
        Q = S_structure(new_dict)

        if self._loop_count % 100 == 0:
            print("Loop count: %d" % self._loop_count)
        self._loop_count += 1

        for encodeX, X in C:
            if len(X["tree"].vertices) >= self._max_num_vertices:
                Q.append(encodeX, X)
                continue

            S = self._expand_1node(X, encodeX)
            S = sub_S_structure(S, R)

            if S.get_length() == 0:
                Q.append(encodeX, X)
                continue

            U, V = self._generic_tree_explorer(S, R)

            for key, value in U:
                Q.append(key, value)

            R.append(encodeX, X)
            for key, value in V:
                R.append(key, value)

        return Q, R

    def mineMFG(self):
        C = S_structure(input_data=self._build_1edge_tree())
        print("Number of Frequent Edges: ", len(C.data))
        for k, v in C:
            print(k, v["freq"])

        new_dict = dict()
        R = S_structure(new_dict)

        M, S = self._generic_tree_explorer(C, R)
        return M
