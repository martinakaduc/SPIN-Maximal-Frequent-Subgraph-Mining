from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import copy
import itertools

import numpy as np
from typing import List
from functools import reduce
from algorithm import encodeGraph

from graph import Graph
from graph import AUTO_EDGE_ID
from graph import Graph
from graph import VACANT_GRAPH_ID
from graph import VACANT_VERTEX_LABEL

class S_structure():
    def __init__(self, input_data=None):
        self.list_encode = list()
        if not input_data:
            self.data = list()
        else:
            self.data = input_data
            for x in input_data:
                self.list_encode.append(x.getEncode())

    def add_graph(self, tree, freq):
        self.data.append(SG_container(tree, freq))
        self.list_encode.append(self.data[-1].getEncode())

    def __iter__(self):
        return iter(self.data)

    def append(self, sg):
        self.data.append(sg)
        self.list_encode.append(self.data[-1].getEncode())

    def get_length(self):
        return len(self.data)

    def get_list_encode(self):
        return self.list_encode

    def get_sg(self, encode):
        return list(filter(lambda x: x.getEncode() == encode, self.data))

def sub_S_structure(here, other):
    result = S_structure()
    other_keys = other.get_list_encode()

    for x in here:
        if x.getEncode() not in other_keys:
            result.append(x)

    return result

def join_S_structure(here, other):
    here_keys = here.get_list_encode()

    for x in other:
        if x.getEncode() not in here_keys:
            here.append(x)

    return here

class SG_container():
    def __init__(self, tree, freq, forFreqEdge=False):
        self.tree = tree
        self.freq = {k: list(v) for k, v in freq.items()}
        self.freq_by_node = {x:{k: v[x] for k, v in freq.items()} for x in self.tree.vertices}
        self.forFreqEdge = forFreqEdge
        self.encode = None
        # if forFreqEdge:
        #     self.encode = self.tree.get_cannonical_tree()
        # else:
        #     self.encode = self.build_encode()

    def get_freq_vertice(self, vid):
        return self.tree.vertices[vid].vlb, self.freq_by_node[vid]

    def get_id_by_lb(self, vlb):
        if vlb in self.tree.set_of_vlb:
            return self.tree.set_of_vlb[vlb]
        else:
            return None

    def getEncode(self):
        if self.encode == None:
            if self.forFreqEdge:
                self.encode = self.tree.get_cannonical_tree()
            else:
                self.encode = self.build_encode()

        return self.encode

    def add_edge_node(self, vid, elb, t_vlb, t_freq):
        new_vid = self.tree.get_num_vertices()
        self.tree.add_vertex(new_vid, t_vlb)
        self.tree.add_edge(AUTO_EDGE_ID, vid, new_vid, elb)

        intersect_graph = sorted(list(set(self.freq.keys()) & set(t_freq.keys())))
        new_freq = {}
        for gid in intersect_graph:
            new_freq[gid] = self.freq[gid] + [t_freq[gid]]

        self.freq = new_freq
        self.freq_by_node[new_vid] = {gid:t_freq[gid] for gid in intersect_graph}

        self.encode = None

    def add_edge(self, vid1, elb, vid2, freq):
        self.tree.add_edge(AUTO_EDGE_ID, vid1, vid2, elb)
        self.freq = {k: v for k, v in self.freq.items() if k in freq}
        self.freq_by_node = {x:{k: v[x] for k, v in self.freq.items()} for x in self.tree.vertices}

        self.encode = None

    def check_vertice(self, lb, freq, min_sup=None):
        vid_found = None
        freq_found = freq

        for vid in self.freq_by_node:
            if self.tree.vertices[vid].vlb != lb:
                continue

            new_freq = dict(self.freq_by_node[vid].items() & freq.items())
            if len(new_freq) >= min_sup:
                vid_found = vid
                freq_found = new_freq

        return vid_found, freq_found

    def check_merge_edge(self, edge, min_sup):
        list_vertice = []
        for x in edge.tree.vertices:
            vlb, v_freq = edge.get_freq_vertice(x)
            vid_found, freq_found = self.check_vertice(vlb, v_freq, min_sup=min_sup)

            if vid_found == None:
                return False

            list_vertice.append((vid_found, freq_found))

        in_elb = edge.tree.vertices[0].edges[1].elb
        cur_elb = self.tree.vertices[list_vertice[0][0]].edges
        # if in_elb == '107':
        #     print("FUCK YOU")
        if not list_vertice[1][0] in cur_elb:
            return (list_vertice[0][0], in_elb, list_vertice[1][0]), tuple(list_vertice[0][1].keys())

        return False

    def build_encode(self):
        order_vid = self.tree.get_cannonical_order()
        encode = ""
        for vid in order_vid:
            encode += str(self.freq_by_node[vid]) + '$'
        return encode

    def change_encode_cannonical(self):
        self.encode = self.tree.get_cannonical_tree()

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
        self._frequent_size1_subgraphs = list()
        self._frequent_subgraphs = dict()
        self._frequent_edges = None
        self._counter = itertools.count()
        self._verbose = verbose
        self._loop_count = 0

        if self._max_num_vertices < self._min_num_vertices:
            print('Max number of vertices can not be smaller than '
                  'min number of that.\n'
                  'Set max_num_vertices = min_num_vertices.')
            self._max_num_vertices = self._min_num_vertices

        self._read_graphs()
        print("Successfully read graph dataset!")

        self._build_1edge_tree()

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
        __freq_edge = []

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

                    if (vid1, vid2) not in vevlb_dict[(vlb1, e.elb, vlb2)][g.gid] and (vid2, vid1) not in vevlb_dict[(vlb1, e.elb, vlb2)][g.gid]:
                        vevlb_dict[(vlb1, e.elb, vlb2)][g.gid].append((vid1, vid2))

        for vevlb, cnt in vevlb_counter.items():
            if cnt >= self._min_support:
                g = Graph(gid=next(self._counter),
                                  is_undirected=self._is_undirected)
                g.add_vertex(0, vevlb[0])
                g.add_vertex(1, vevlb[2])
                g.add_edge(AUTO_EDGE_ID, 0, 1, vevlb[1])

                # Mark edge as frequent
                for gid in vevlb_dict[vevlb]:
                    for pair in vevlb_dict[vevlb][gid]:
                        self.graphs[gid].set_freq_edge(pair[0], pair[1])

                # Append into set
                new_tree = SG_container(g, vevlb_dict[vevlb])
                self._frequent_size1_subgraphs.append(new_tree)
                __freq_edge.append(SG_container(g, vevlb_dict[vevlb], forFreqEdge=True))

                # if self._min_num_vertices <= 1:
                #     self._report_size1(g, support=cnt)

        # self._frequent_edges = S_structure(input_data=self._frequent_size1_subgraphs)
        self._frequent_edges = S_structure(input_data=__freq_edge)

        # print(self._frequent_size1_subgraphs)
        if self._min_num_vertices > 1:
            self._counter = itertools.count()

    def _expand_1node(self, lc_graph):
        # print("=========================")
        # print("Expand: ", lc_graph.encode)
        result = S_structure()

        # Expand & recalculate which graph hold
        lc_graph_vertices = list(lc_graph.tree.vertices.keys())
        for vid in lc_graph_vertices:
            vlb, v_freq = lc_graph.get_freq_vertice(vid)
            freq_edge_set = set()

            for gid, _vid in v_freq.items():
                freq_edge_set = freq_edge_set | self.graphs[gid].get_freq_edges(_vid)

            # print("Freq Edge Set: ", freq_edge_set)
            for edge_enc in freq_edge_set:
                can_edges = self._frequent_edges.get_sg(edge_enc)
                # print(can_edges)
                for edge in can_edges:
                    vid_in_edge, _ = edge.check_vertice(vlb, v_freq, min_sup=self._min_support)
                    # print(vid_in_edge)
                    if vid_in_edge != None:
                        vid_target_in_edge = not vid_in_edge
                        vlb_target, v_target_freq = edge.get_freq_vertice(vid_target_in_edge)

                        in_lc_graph, inter_freq = lc_graph.check_vertice(vlb_target, v_target_freq, min_sup=self._min_support)
                        if in_lc_graph == None:
                            # Add this edge to new graph
                            new_tree = copy.deepcopy(lc_graph)
                            elb_target = edge.tree.vertices[vid_in_edge].edges[vid_target_in_edge].elb
                            new_tree.add_edge_node(vid, elb_target, vlb_target, inter_freq)
                            # print(new_tree.encode)
                            if new_tree.getEncode() > lc_graph.getEncode():
                                result.append(new_tree)

        return result

    def _expand_subgraph(self, tree):
        list_candidate_edge = []
        for edge in self._frequent_edges:
            chk_res = tree.check_merge_edge(edge, self._min_support)
            if chk_res != False:
                list_candidate_edge.append(chk_res)

        # print(list_candidate_edge)
        if not list_candidate_edge:
            return S_structure()

        S = self._search_graph(copy.deepcopy(tree), list_candidate_edge)

        # print(S.data)
        # Find maximal

        return S

    def _search_graph(self, tree, list_edges):
        list_freq =set(reduce(lambda x,y: x+[y[1]], list_edges, list()))
        Q = S_structure()

        for freq in list_freq:
            new_list_edge = []
            new_tree = copy.deepcopy(tree)
            for edge, edge_freq in list_edges:
                if edge_freq != freq:
                    new_list_edge.append((edge, edge_freq))
                else:
                    new_tree.add_edge(edge[0], edge[1], edge[2], edge_freq)

            SG = self._search_graph(new_tree, new_list_edge)
            if SG.get_length() == 0:
                Q.append(new_tree)
            else:
                for sg in SG:
                    if sg.encode not in Q.get_list_encode():
                        Q.append(sg)

        return Q

    def _generic_tree_explorer(self, C, R):
        Q = S_structure()

        if self._loop_count % 100 == 0:
            print("Loop count: %d" % self._loop_count)
        self._loop_count += 1

        for X in C:
            if len(X.tree.vertices) >= self._max_num_vertices:
                X_expand = self._expand_subgraph(copy.deepcopy(X))
                if X_expand.get_length() > 0:
                    for x in X_expand:
                        Q.append(x)
                else:
                    Q.append(X)
                continue

            S = self._expand_1node(X)
            if S.get_length() == 0:
                X_expand = self._expand_subgraph(copy.deepcopy(X))
                if X_expand.get_length() > 0:
                    for x in X_expand:
                        Q.append(x)
                else:
                    Q.append(X)
                continue

            S = sub_S_structure(S, R)

            if S.get_length() == 0:
                continue

            U, V = self._generic_tree_explorer(S, R)
            for x in U:
                Q.append(x)

            R.append(X)
            # for x in V:
            #     R.append(x)

        return Q, R

    def mineMFG(self):
        C = S_structure(input_data=self._frequent_size1_subgraphs)
        print("Number of Frequent Edges: ", len(C.data))
        # for x in C:
        #     print(x.tree.get_cannonical_tree(), x.freq)

        R = S_structure()

        M, S = self._generic_tree_explorer(C, R)
        return M
