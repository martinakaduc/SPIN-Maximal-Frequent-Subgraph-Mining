#!/usr/bin/python3
# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from algorithm import *
from spin3 import *
from utils import *
from config import parser

import time
import os
import sys

def main(FLAGS=None):
    """Run gSpan."""

    if FLAGS is None:
        FLAGS, _ = parser.parse_known_args(args=sys.argv[1:])

    if not os.path.exists(FLAGS.database_file_name):
        print('{} does not exist.'.format(FLAGS.database_file_name))
        sys.exit()

    spin = SPIN(
        database_file_name=FLAGS.database_file_name,
        min_support=FLAGS.min_support,
        min_num_vertices=FLAGS.lower_bound_of_num_vertices,
        max_num_vertices=FLAGS.upper_bound_of_num_vertices,
        max_ngraphs=FLAGS.num_graphs,
        is_undirected=(not FLAGS.directed),
        verbose=FLAGS.verbose,
        visualize=FLAGS.plot,
        where=FLAGS.where,
        do_optimize=FLAGS.optimize
    )

    spin.mineMFG()
    spin.time_stats()
    return spin

if __name__ == '__main__':
    main()
    # To run: python main.py -s [support] [datafile]

# if __name__ == '__main__':
#     spin = SPIN(FILENAME,
#                     min_support = THETA,
#                     max_num_vertices = MAX_NUM_VERTICE,
#                     max_ngraphs = MAX_NGRAPHS)
#     time_start = time.time()
#     M = spin.mineMFG()
#     time_end = time.time()
#
#     print("Time: %.5f" % (time_end-time_start))
#
#     # set_pattern = set()
#     # for m in M:
#     #     # print(m.encode, list(m.freq.keys()))
#     #     set_pattern.add(tuple(sorted(list(m.freq.keys()))))
#     #
#     # max_pattern = dict()
#     # pat_m = dict()
#     # for pat in set_pattern:
#     #     # print(pat)
#     #     max_pattern[pat] = "$#"
#     #     for m in M:
#     #         gt = tuple(sorted(list(m.freq.keys())))
#     #         m.change_encode_cannonical()
#     #         if pat == gt:
#     #             if len(m.encode) > len(max_pattern[pat]) or (len(m.encode) == len(max_pattern[pat]) and m.encode > max_pattern[pat]):
#     #                 max_pattern[pat] = m.encode
#     #                 pat_m[pat] = m
#     #
#     # for gt, pat in max_pattern.items():
#     #     # print(gt, pat)
#     #     print(gt)
#     #     pat_m[gt].tree.display()
