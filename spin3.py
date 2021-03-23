from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import itertools

import numpy as np
import pandas as pd
from functools import reduce
from algorithm import encodeGraph
import time

from graph import Graph
from graph import AUTO_EDGE_ID
from graph import VACANT_GRAPH_ID
from graph import VACANT_VERTEX_LABEL

def record_timestamp(func):
    """Record timestamp before and after call of `func`."""
    def deco(self):
        self.timestamps[func.__name__ + '_in'] = time.time()
        func(self)
        self.timestamps[func.__name__ + '_out'] = time.time()
    return deco

class DFSedge(object):
    """DFSedge class."""

    def __init__(self, frm, to, vevlb):
        """Initialize DFSedge instance."""
        self.frm = frm
        self.to = to
        self.vevlb = vevlb

    def __eq__(self, other):
        """Check equivalence of DFSedge."""
        return (self.frm == other.frm and
                self.to == other.to and
                self.vevlb == other.vevlb)

    def __ne__(self, other):
        """Check if not equal."""
        return not self.__eq__(other)

    def __repr__(self):
        """Represent DFScode in string way."""
        return '(frm={}, to={}, vevlb={})'.format(
            self.frm, self.to, self.vevlb
        )


class DFScode(list):
    """DFScode is a list of DFSedge."""

    def __init__(self):
        """Initialize DFScode."""
        self.rmpath = list()

    def __eq__(self, other):
        """Check equivalence of DFScode."""
        la, lb = len(self), len(other)
        if la != lb:
            return False
        for i in range(la):
            if self[i] != other[i]:
                return False
        return True

    def __ne__(self, other):
        """Check if not equal."""
        return not self.__eq__(other)

    def print_embed(self):
        return ''.join(['[', ','.join(
            [str(dfsedge) for dfsedge in self]), ']']
        )

    def __repr__(self):
        """Represent DFScode in string way."""
        g = self._construct_graph()
        cannonical = g.get_cannonical_tree()

        return cannonical

    def represent(self, graph, projected):
        history = History(projected)
        list_vid = list(history.vertices_used.keys())
        sorted_vid = tuple(sorted(list_vid))
        return sorted_vid

    def push_back(self, frm, to, vevlb):
        """Update DFScode by adding one edge."""
        self.append(DFSedge(frm, to, vevlb))
        return self

    def _construct_graph(self, gid=VACANT_GRAPH_ID, is_undirected=True):
        g = Graph(gid,
                  is_undirected=is_undirected,
                  eid_auto_increment=True)
        for dfsedge in self:
            frm, to, (vlb1, elb, vlb2) = dfsedge.frm, dfsedge.to, dfsedge.vevlb
            if vlb1 != VACANT_VERTEX_LABEL:
                g.add_vertex(frm, vlb1)
            if vlb2 != VACANT_VERTEX_LABEL:
                g.add_vertex(to, vlb2)
            g.add_edge(AUTO_EDGE_ID, frm, to, elb)

        return g

    def to_graph(self, gid=VACANT_GRAPH_ID, is_undirected=True):
        """Construct a graph according to the dfs code."""
        g = self._construct_graph(gid=gid, is_undirected=is_undirected)
        return g

    def from_graph(self, g):
        """Build DFScode from graph `g`."""
        raise NotImplementedError('Not inplemented yet.')

    def get_num_vertices(self):
        """Return number of vertices in the corresponding graph."""
        return len(set(
            [dfsedge.frm for dfsedge in self] +
            [dfsedge.to for dfsedge in self]
        ))

class PDFS(object):
    """PDFS class."""

    def __init__(self, gid=VACANT_GRAPH_ID, edge=None, prev=None):
        """Initialize PDFS instance."""
        self.gid = gid
        self.edge = edge
        self.prev = prev

    def check_duplicate(self, other):
        return (self.gid == other.gid and
                self.edge.to == other.edge.to)

class Projected(list):
    """Projected is a list of PDFS.

    Each element of Projected is a projection one frequent graph in one
    original graph.
    """

    def __init__(self):
        """Initialize Projected instance."""
        super(Projected, self).__init__()

    def push_back(self, gid, edge, prev):
        """Update this Projected instance."""
        self.append(PDFS(gid, edge, prev))
        return self

class History(object):
    """History class."""

    def __init__(self, pdfs, dfs_code=False):
        """Initialize History instance."""
        super(History, self).__init__()
        self.edges = list()
        self.vertices_used = collections.defaultdict(lambda:-1)
        self.edges_used = collections.defaultdict(lambda:-1)
        self.pdfs = pdfs

        if pdfs is None:
            return

        if not dfs_code:
            while pdfs:
                e = pdfs.edge
                self.edges.append(e)
                (self.vertices_used[e.frm],
                 self.vertices_used[e.to],
                 self.edges_used[e.eid]) = 1, 1, 1

                pdfs = pdfs.prev
        else:
            dfs_i = -1
            while pdfs:
                e = pdfs.edge
                self.edges.append(e)
                self.vertices_used[e.frm] = dfs_code[dfs_i].frm
                self.vertices_used[e.to] = dfs_code[dfs_i].to
                self.edges_used[e.eid] = 1

                dfs_i -= 1
                pdfs = pdfs.prev

        self.edges = self.edges[::-1]

    def has_vertex(self, vid):
        """Check if the vertex with vid exists in the history."""
        return self.vertices_used[vid] != -1

    def has_edge(self, eid):
        """Check if the edge with eid exists in the history."""
        return self.edges_used[eid] != -1

    def get_vertex_mapping(self, vid):
        return self.vertices_used[vid]

class SPIN(object):
    """`SPIN` algorithm."""

    def __init__(self,
                 database_file_name,
                 min_support=10,
                 min_num_vertices=2,
                 max_num_vertices=float('inf'),
                 max_ngraphs=float('inf'),
                 is_undirected=True,
                 verbose=False,
                 visualize=False,
                 where=False):
        """Initialize gSpan instance."""
        self._database_file_name = database_file_name
        self.graphs = dict()
        self._max_ngraphs = max_ngraphs
        self._is_undirected = is_undirected
        self._min_support = min_support
        self._min_num_vertices = min_num_vertices
        self._max_num_vertices = max_num_vertices
        self._DFScode = DFScode()
        self._support = 0
        self._frequent_size1_subgraphs = list()
        # Include subgraphs with
        # any num(but >= 2, <= max_num_vertices) of vertices.
        self._frequent_subgraphs = list()
        self._counter = itertools.count()
        self._verbose = verbose
        self._visualize = visualize
        self._where = where
        self.timestamps = dict()
        self._loop_count = 0
        if self._max_num_vertices < self._min_num_vertices:
            print('Max number of vertices can not be smaller than '
                  'min number of that.\n'
                  'Set max_num_vertices = min_num_vertices.')
            self._max_num_vertices = self._min_num_vertices
        self._report_df = pd.DataFrame()

    def time_stats(self):
        """Print stats of time."""
        func_names = ['_read_graphs', 'mineMFG']
        time_deltas = collections.defaultdict(float)
        for fn in func_names:
            time_deltas[fn] = self.timestamps[fn + '_out'] - self.timestamps[fn + '_in']

        print("Read:\t%.5f s" % time_deltas['_read_graphs'])
        print("Mine:\t%.5f s" % (time_deltas['mineMFG'] - time_deltas['_read_graphs']))
        print("Total:\t%.5f s" % time_deltas['mineMFG'])

        return self

    @record_timestamp
    def _read_graphs(self):
        self.graphs = dict()
        with open(self._database_file_name, 'r', encoding='utf-8') as f:
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

    def _generate_1edge_frequent_subgraphs(self):
        vevlb_counter = collections.Counter()
        vevlb_counted = set()
        vevlb_dict = dict()
        C = collections.defaultdict(Projected)

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

                    if (vid1, vid2) not in vevlb_dict[(vlb1, e.elb, vlb2)][g.gid]:
                        vevlb_dict[(vlb1, e.elb, vlb2)][g.gid].append((vid1, vid2))

        # add frequent vertices.
        for vevlb, cnt in vevlb_counter.items():
            if cnt >= self._min_support:
                g = Graph(gid=next(self._counter),
                                  is_undirected=self._is_undirected)
                g.add_vertex(0, vevlb[0])
                g.add_vertex(1, vevlb[2])
                g.add_edge(AUTO_EDGE_ID, 0, 1, vevlb[1])

                for g_key in self.graphs:
                    if self.graphs[g_key].gid in vevlb_dict[vevlb]:
                        for pair in vevlb_dict[vevlb][self.graphs[g_key].gid]:
                            e = self.graphs[g_key].set_freq_edge(pair[0], pair[1])
                            C[vevlb].append(PDFS(self.graphs[g_key].gid, e, None))

                self._frequent_size1_subgraphs.append(g)
                if self._min_num_vertices <= 1:
                    self._report_size1(g, support=cnt)
            else:
                continue
        if self._min_num_vertices > 1:
            self._counter = itertools.count()

        return C

    def _get_support(self, projected):
        return len(set([pdfs.gid for pdfs in projected]))

    def _report_size1(self, g, support):
        g.display()
        print('\nSupport: {}'.format(support))
        print('\n-----------------\n')

    def _report(self, projected):
        self._frequent_subgraphs.append(copy.copy(self._DFScode))
        if self._DFScode.get_num_vertices() < self._min_num_vertices:
            return
        g = self._DFScode.to_graph(gid=next(self._counter),
                                   is_undirected=self._is_undirected)
        display_str = g.display()
        # print('\nCannonical: {}'.format(self._DFScode.represent(self.graphs[projected[0].gid], projected[0])))
        print('\nSupport: {}'.format(self._support))

        # Add some report info to pandas dataframe "self._report_df".
        self._report_df = self._report_df.append(
            pd.DataFrame(
                {
                    'support': [self._support],
                    'description': [display_str],
                    'num_vert': self._DFScode.get_num_vertices()
                },
                index=[int(repr(self._counter)[6:-1])]
            )
        )
        if self._visualize:
            g.plot()
        if self._where:
            print('where: {}'.format(list(set([p.gid for p in projected]))))
        print('\n-----------------\n')

    def _expand_1node_start(self, projected):
        # self._support = self._get_support(projected)
        # self._report(projected)

        num_vertices = self._DFScode.get_num_vertices()
        maxtoc = self._DFScode[-1].to

        forward_root = collections.defaultdict(Projected)
        for p in projected:
            g = self.graphs[p.gid]
            history = History(p)

            if num_vertices >= self._max_num_vertices:
                continue

            edges = self._get_forward_pure_edges(g,
                                                 p.edge,
                                                 history)
            for e in edges:
                forward_root[
                    (maxtoc, e.elb, g.vertices[e.to].vlb)
                ].append(PDFS(g.gid, e, p))

        maxtoc -= 1
        for p in copy.deepcopy(projected):
            g = self.graphs[p.gid]
            history = History(p)

            if num_vertices >= self._max_num_vertices:
                continue

            p.edge.frm, p.edge.to = p.edge.to, p.edge.frm
            edges = self._get_forward_pure_edges(g,
                                                 p.edge,
                                                 history)
            for e in edges:
                forward_root[
                    (maxtoc, e.elb, g.vertices[e.to].vlb)
                ].append(PDFS(g.gid, e, p))

        unfreq_edge = []
        for frm, elb, vlb2 in forward_root:
            support = self._get_support(forward_root[(frm, elb, vlb2)])
            if support < self._min_support:
                unfreq_edge.append((frm, elb, vlb2))

        for e in unfreq_edge:
            del forward_root[e]

        return forward_root

    def _get_forward_pure_edges(self, g, rm_edge, history):
        result = []
        for to, e in g.vertices[rm_edge.to].edges.items():
            if e.is_freq and (not history.has_vertex(e.to)):
                result.append(e)
        return result

    def _expand_1node(self, projected, prev_cand_edge=None):
        # self._support = self._get_support(projected)
        # self._report(projected)

        num_vertices = self._DFScode.get_num_vertices()
        maxtoc = self._DFScode[-1].to

        forward_root = collections.defaultdict(Projected)
        for p in projected:
            g = self.graphs[p.gid]
            history = History(p)

            if num_vertices >= self._max_num_vertices:
                continue

            edges = self._get_forward_pure_edges(g,
                                                 p.edge,
                                                 history)
            for e in edges:
                forward_root[
                    (maxtoc, e.elb, g.vertices[e.to].vlb)
                ].append(PDFS(g.gid, e, p))

        unfreq_edge = []
        for frm, elb, vlb2 in forward_root:
            support = self._get_support(forward_root[(frm, elb, vlb2)])
            if support < self._min_support:
                unfreq_edge.append((frm, elb, vlb2))

        for e in unfreq_edge:
            del forward_root[e]

        if prev_cand_edge != None:
            unfreq_edge = []
            list_projected_gid = [x.gid for x in projected]
            list_history = collections.defaultdict(list)

            for proj in projected:
                list_history[proj.gid].append(History(proj))

            for frm, elb, vlb2 in prev_cand_edge:
                new_embedding = list(filter(lambda x: x.gid in list_projected_gid, prev_cand_edge[(frm, elb, vlb2)]))
                support = self._get_support(new_embedding)

                if support < self._min_support:
                    unfreq_edge.append((frm, elb, vlb2))
                    continue
                else:
                    duplicate_pdfs = []
                    duplicate_pdfs_idx = []
                    for i, pdfs in enumerate(new_embedding):
                        for history in list_history[pdfs.gid]:
                            if history.has_vertex(pdfs.edge.frm):
                                if history.has_vertex(pdfs.edge.to):
                                    duplicate_pdfs.append(pdfs)
                                    duplicate_pdfs_idx.append(i)
                                else:
                                    new_embedding[i].prev = history.pdfs
                                break

                    if support - self._get_support(duplicate_pdfs) < self._min_support:
                        unfreq_edge.append((frm, elb, vlb2))
                        continue
                    else:
                        for idx in sorted(duplicate_pdfs_idx, reverse=True):
                            del new_embedding[idx]

                prev_cand_edge[(frm, elb, vlb2)] = new_embedding

            for e in unfreq_edge:
                del prev_cand_edge[e]

            for frm, elb, vlb2 in forward_root:
                prev_cand_edge[(frm, elb, vlb2)] = forward_root[(frm, elb, vlb2)]

            return prev_cand_edge

        else:
            return forward_root

    def _remove_duplicate(self, pre_S, R):
        duplicate = []
        maxtoc = self._DFScode[-1].to

        for frm, elb, vlb2 in pre_S:
            self._DFScode.push_back(
                frm, maxtoc + 1,
                (VACANT_VERTEX_LABEL, elb, vlb2)
            )

            cannonical = self._get_all_embedding(pre_S[(frm, elb, vlb2)])
            for r in R:
                intersect = cannonical & r
                if len(intersect) >= self._min_support:
                    duplicate.append((frm, elb, vlb2))
                    break

            self._DFScode.pop()

        for e in duplicate:
            del pre_S[e]

        return pre_S

    def _get_all_embedding(self, pdfs):
        return frozenset(map(lambda p: (p.gid, self._DFScode.represent(self.graphs[p.gid], p)), pdfs))

    def _generic_tree_explorer(self, C, R):
        if self._loop_count % 100 == 0:
            print("Loop count: %d\n" % self._loop_count)
        self._loop_count += 1

        maxtoc = self._DFScode[-1].to
        for fevlb, projected in C.items():
            # Check current DFS has cand fevlb
            self._DFScode.push_back(
                fevlb[0], maxtoc + 1,
                (VACANT_VERTEX_LABEL, fevlb[1], fevlb[2])
            )

            prev_cand_edge = copy.deepcopy(C)
            del prev_cand_edge[fevlb]

            pre_S = self._expand_1node(projected, prev_cand_edge)

            S = self._remove_duplicate(copy.deepcopy(pre_S), R)
            if (len(S) == len(pre_S)):
                _, V = self._generic_tree_explorer(S, R)
            else:
                V = []

            if len(pre_S) == 0 and not self._check_external_assoc_edge(projected):
                if not self._get_all_embedding(projected) in R:
                    self._maximal_expand(projected)

            R = list(set(R + [self._get_all_embedding(projected)] + V))

            self._DFScode.pop()

        return None, R

    def _generic_tree_explorer_start(self, C, R):
        self._loop_count = 1

        for vevlb, projected in C.items():
            self._DFScode.push_back(0, 1, vevlb)
            pre_S = self._expand_1node_start(projected)

            S = self._remove_duplicate(pre_S, R)
            _, V = self._generic_tree_explorer(S, R)

            if len(pre_S) == 0 and not self._check_external_assoc_edge(projected):
                if not self._get_all_embedding(projected) in R:
                    self._maximal_expand(projected)

            R = list(set(R + [self._get_all_embedding(projected)] + V))

            self._DFScode.pop()

        return None, R

    def _get_external_associative_edges(self, g, rm_edge, history):
        result = []
        for to, e in g.vertices[rm_edge.to].edges.items():
            if e.is_freq and (not history.has_vertex(e.to)):
                result.append(e)
        return result

    def _check_external_assoc_edge(self, projected):
        external_assoc_edges = collections.defaultdict(Projected)
        maxtoc = self._DFScode[-1].to

        for p in projected:
            g = self.graphs[p.gid]
            history = History(p)
            current_vid = maxtoc

            while True:
                edges = self._get_external_associative_edges(g, p.edge, history)
                for e in edges:
                    external_assoc_edges[
                        (current_vid, e.elb, g.vertices[e.to].vlb)
                    ].append(PDFS(g.gid, e, None))

                current_vid -= 1
                if p.prev:
                    p = p.prev
                else:
                    for to, e in g.vertices[p.edge.frm].edges.items():
                        if e.is_freq and (not history.has_vertex(e.to)):
                            external_assoc_edges[
                                (current_vid, e.elb, g.vertices[e.to].vlb)
                            ].append(PDFS(g.gid, e, None))
                    break

        for e in external_assoc_edges:
            if self._get_support(external_assoc_edges[e]) >= self._min_support:
                return True

        return False

    def _get_internal_associative_edges(self, g, rm_edge, history):
        result = []
        for to, e in g.vertices[rm_edge.to].edges.items():
            if e.is_freq and (not history.has_edge(e.eid)) and history.has_vertex(e.to):
                result.append(e)
        return result

    def _maximal_expand(self, projected):
        candidate_edges = collections.defaultdict(Projected)
        for p in projected:
            g = self.graphs[p.gid]
            history = History(p, dfs_code=self._DFScode)

            while True:
                edges = self._get_internal_associative_edges(g, p.edge, history)
                for e in edges:
                    candidate_edges[
                        (history.get_vertex_mapping(e.frm), e.elb, history.get_vertex_mapping(e.to))
                    ].append(PDFS(g.gid, e, p))

                if p.prev:
                    p = p.prev
                else:
                    for to, e in g.vertices[p.edge.frm].edges.items():
                        if e.is_freq and (not history.has_edge(e.eid)) and history.has_vertex(e.to):
                            candidate_edges[
                                (history.get_vertex_mapping(e.frm), e.elb, history.get_vertex_mapping(e.to))
                            ].append(PDFS(g.gid, e, p))
                    break

        unfreq_edge = []
        for e in candidate_edges:
            if self._get_support(candidate_edges[e]) < self._min_support:
                unfreq_edge.append(e)

        for e in unfreq_edge:
            del candidate_edges[e]

        self._search_graph(candidate_edges, [p.gid for p in projected])

    def _search_graph(self, candidate_edges, prev_embedding):
        # Try 2 add all edges
        join_embed = set(prev_embedding)
        join_embed = list(reduce(lambda x, p: x & set([pdfs.gid for pdfs in p[1]]), candidate_edges.items(), join_embed))

        # Bottom-up pruning
        if len(join_embed) >= self._min_support:
            for e in candidate_edges:
                self._DFScode.push_back(
                    e[0], e[2],
                    (VACANT_VERTEX_LABEL, e[1], VACANT_VERTEX_LABEL)
                )

            temp_projected = []
            for embed in join_embed:
                temp_projected.append(PDFS(embed))

            self._support = len(join_embed)
            self._report(temp_projected)

            for _ in candidate_edges:
                self._DFScode.pop()

            return

        for e in candidate_edges:
            self._DFScode.push_back(
                e[0], e[2],
                (VACANT_VERTEX_LABEL, e[1], VACANT_VERTEX_LABEL)
            )

            new_candidate_edges = copy.deepcopy(candidate_edges)
            del new_candidate_edges[e]

            # Update embeding
            current_embeding = [x.gid for x in candidate_edges[e]]
            unfreq_edge = []
            for nce in new_candidate_edges:
                filtered_embedding = list(filter(lambda x: x.gid in current_embeding, new_candidate_edges[nce]))
                if self._get_support(filtered_embedding) >= self._min_support:
                    new_candidate_edges[nce] = filtered_embedding
                else:
                    unfreq_edge.append(nce)

            for nce in unfreq_edge:
                del new_candidate_edges[nce]

            self._search_graph(new_candidate_edges, current_embeding)
            self._DFScode.pop()

    @record_timestamp
    def mineMFG(self):
        self._read_graphs()
        C = self._generate_1edge_frequent_subgraphs()
        R = []
        M, S = self._generic_tree_explorer_start(C, R)
        # return self._frequent_subgraphs
