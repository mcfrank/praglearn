from nips import *
from parallel_dialogues import *

mean_P_understood = {}
for name, pattern in [("flat2", "par/flat-50-2-*.pickle"),
                      ("flat3", "par/flat-50-3-*.pickle"),
                      ("horn", "par/horn-50-*.pickle"),
                      ]:
    mean_P_understood[name] = np.mean([extract_P_understood(pattern, i)
                                       for i in xrange(50)])
