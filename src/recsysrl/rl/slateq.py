import numpy as np


def slateq_value(item_q_values, slate):
    return float(np.sum(np.take(item_q_values, slate)))
