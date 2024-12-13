import sys
import numpy as np
import struct

LABEL_UNSIGNED = 'I'
LABEL_FLOAT = 'f'

if (len(sys.argv) == 9) :
    path = sys.argv[1]

    min_B = int(sys.argv[2])
    max_B = int(sys.argv[3])
    B = np.random.randint(min_B, max_B+1) # As upper bound is excluded in numpy

    C = int(sys.argv[4])
    H = int(sys.argv[5])
    W = int(sys.argv[6])

    min_val = float(sys.argv[7])
    max_val = float(sys.argv[8])

    ba = bytearray(b'CPIC')
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, 4)) ) # As 4 is tensor rank, i.e. pytorch's tensor.dim()
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, B)) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, C)) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, H)) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, W)) )

    data = np.random.rand(B*C*H*W)
    for el in data :
        el = el * (max_val-min_val) + min_val
        ba.extend( bytearray(struct.pack(LABEL_FLOAT, el)) )

    file = open(path, 'wb')
    file.write(ba)
    file.close()

else :
    print('Usage: random_cpic <path> <min_B> <max_B> <C> <H> <W> <min_val> <max_val>')
