from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from easydict import EasyDict as easydict

def parse_config_file(path_to_config):

    with open(path_to_config) as f:
        cfg = yaml.load(f)

    return easydict(cfg)


def merge_a_into_b(a, b):
    '''
    Merge config directory a into config dictionary b, clobbering the
    options in b whenever they are also specific in a.
    '''
    if type(a) is not easydict:
        return

    for k, v in a.iteritems():
        # a must specific keys that are in b
        if not b.has_key(k):
            raise KeyError("{} is not a valid config key".format(k))

        # the type must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))
        # recursively merge dictionaries
        if type(v) is easydict:
            try:
                merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v
