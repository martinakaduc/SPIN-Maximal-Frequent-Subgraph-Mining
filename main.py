from algorithm import *
from spin2 import *
from utils import *
import time

FILENAME = "8graphs_2pattern15nodes.lg"
# FILENAME = "100graphs_10patterns.lg"
THETA = 3
MAX_NUM_VERTICE = float('inf')

if __name__ == '__main__':
    spin = SPIN(FILENAME, min_support = THETA, max_num_vertices = MAX_NUM_VERTICE)
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
