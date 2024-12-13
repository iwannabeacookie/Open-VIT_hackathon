import sys
import numpy as np
import struct

LABEL_UNSIGNED = 'I'
LABEL_FLOAT = 'f'
SIZEOF_UNSIGNED = 4
SIZEOF_FLOAT = 4

if (len(sys.argv) == 2) :
    cprd_path = sys.argv[1]

    # Extract the prediction
    file = open(cprd_path, 'rb')
    ba = bytearray( file.read() )
    file.close()

    cvit_label = ba[0:4]
    ba = ba[4 : ]
    assert cvit_label == b'CPRD'

    b = struct.unpack(LABEL_UNSIGNED, ba[0:SIZEOF_UNSIGNED])
    b = b[0]
    cls = struct.unpack(LABEL_UNSIGNED, ba[SIZEOF_UNSIGNED:2*SIZEOF_UNSIGNED])
    cls = cls[0]
    ba = ba[2*SIZEOF_UNSIGNED : ]

    classes = []
    for i in range (b) :
        el = struct.unpack(LABEL_UNSIGNED, ba[i*SIZEOF_UNSIGNED : (i+1)*SIZEOF_UNSIGNED])
        el = el[0]
        classes.append(el)
    ba = ba[SIZEOF_UNSIGNED*b : ]

    prob = []
    for i in range (b) :
        el = struct.unpack(LABEL_FLOAT, ba[i*SIZEOF_FLOAT : (i+1)*SIZEOF_FLOAT])
        el = el[0]
        prob.append(el)
    ba = ba[SIZEOF_FLOAT*b : ]

    # Eventually read the probability matrix
    # prob_matrix = []
    # for i in range (b*cls) :
    #     el = struct.unpack(LABEL_FLOAT, ba[i*SIZEOF_FLOAT : (i+1)*SIZEOF_FLOAT])
    #     el = el[0]
    #     prob_matrix.append(el)
    # ba = ba[SIZEOF_FLOAT*b*cls : ]



    # Plot the prediction
    print(f'Prediction batch with {b} images, {cls} possible classes:')

    for i in range(b) :
        print('   ', end='')
        print(f'image[{i}]:', end=' ')
        print(f'class {classes[i]}', end=', ')
        print(f'prob {prob[i]:7.5f}')
    print('')

    # print('   ', end='')
    # print(f'Probability Matrix[{b}x{cls}]:')
    # for i in range(b) :
    #     print('   ', end='')
    #     for j in range(cls) :
    #         print(f'{prob_matrix[i*cls + j]:13.9f}', end=' ')
    #     print('')
    # print('')



else :
    print('Usage: plot_prediction <cprd_path>')
