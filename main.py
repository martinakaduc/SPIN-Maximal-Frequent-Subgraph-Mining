from algorithm import *
from spin3 import *
from utils import *
import time

FILENAME = "COX2_convert.txt"
# FILENAME = "8graphs_2patterns.lg"
THETA = 150
MAX_NUM_VERTICE = float('inf')
MAX_NGRAPHS = float('inf')

if __name__ == '__main__':
    spin = SPIN(FILENAME,
                    min_support = THETA,
                    max_num_vertices = MAX_NUM_VERTICE,
                    max_ngraphs = MAX_NGRAPHS)
    time_start = time.time()
    M = spin.mineMFG()
    time_end = time.time()

    print("Time: %.5f" % (time_end-time_start))

    set_pattern = set()
    for m in M:
        # print(m.encode, list(m.freq.keys()))
        set_pattern.add(tuple(sorted(list(m.freq.keys()))))

    max_pattern = dict()
    pat_m = dict()
    for pat in set_pattern:
        # print(pat)
        max_pattern[pat] = "$#"
        for m in M:
            gt = tuple(sorted(list(m.freq.keys())))
            m.change_encode_cannonical()
            if pat == gt:
                if len(m.encode) > len(max_pattern[pat]) or (len(m.encode) == len(max_pattern[pat]) and m.encode > max_pattern[pat]):
                    max_pattern[pat] = m.encode
                    pat_m[pat] = m

    for gt, pat in max_pattern.items():
        # print(gt, pat)
        print(gt)
        pat_m[gt].tree.display()
