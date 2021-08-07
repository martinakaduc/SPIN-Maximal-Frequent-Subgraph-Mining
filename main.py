#!/usr/bin/python3
# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from spin3 import *
from utils import *
from config import parser

import time
import os
import sys

def main(FLAGS=None):
    """Run SPIN."""

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
        max_time=FLAGS.max_time,
        do_optimize=FLAGS.optimize
    )

    spin.mineMFG()
    spin.time_stats()
    return spin

if __name__ == '__main__':
    main()
