from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
    )
str = type('')

import numpy as np

color_dtype = np.dtype([
    (native_str('red'),   np.uint8),
    (native_str('green'), np.uint8),
    (native_str('blue'),  np.uint8),
    ])

