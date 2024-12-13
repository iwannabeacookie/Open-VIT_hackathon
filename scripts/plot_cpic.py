import sys
import numpy as np
import struct
import matplotlib.pyplot as plt

LABEL_UNSIGNED = 'I'
LABEL_FLOAT = 'f'
SIZEOF_UNSIGNED = 4
SIZEOF_FLOAT = 4

if (len(sys.argv) == 2) :
    path = sys.argv[1]

    # Extract cpic from file
    file = open(path, 'rb')
    ba = bytearray( file.read() )
    file.close()

    cvit_label = ba[0:4]
    ba = ba[4 : ]
    assert cvit_label == b'CPIC'

    dim = struct.unpack(LABEL_UNSIGNED, ba[0:SIZEOF_UNSIGNED])
    dim = dim[0]
    ba = ba[SIZEOF_UNSIGNED : ]

    shape = []
    tot_elements = 1
    for i in range (dim) :
        sh = struct.unpack(LABEL_UNSIGNED, ba[i*SIZEOF_UNSIGNED : (i+1)*SIZEOF_UNSIGNED])
        sh = sh[0]
        tot_elements *= sh
        shape.append(sh)
    ba = ba[SIZEOF_UNSIGNED*dim : ]

    data = []
    for i in range (tot_elements) :
        el = struct.unpack(LABEL_FLOAT, ba[i*SIZEOF_FLOAT : (i+1)*SIZEOF_FLOAT])
        el = el[0]
        data.append(el)
    ba = ba[SIZEOF_FLOAT*tot_elements : ]

    batch = np.array(data)
    batch = np.ndarray(shape, buffer=batch)
    batch_size = shape[0]

    # Plot the tensor
    if (batch_size > 64) :
        print(f'Plotting 64 out of {batch_size} images of the batch {path}')
        batch_size = 64
    else :
        print(f'Plotting all the {batch_size} images of the batch {path}')

    plt.figure(figsize=(8, np.ceil(batch_size/8)))

    for i in range(batch_size) :
        plt.subplot(int(np.ceil(batch_size/8)), 8, i + 1)

        image = np.clip( np.moveaxis(batch[i], 0, -1), a_min=0, a_max=1 )
        plt.imshow(image)

        plt.title(f"Image {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

else :
    print('Usage: plot_cpic <cpic_path>')
