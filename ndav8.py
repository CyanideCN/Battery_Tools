import mmap
import enum

import numpy as np
import pandas as pd

class NDAStepType(enum.IntEnum):
    CC_Chg = 1
    CC_DChg = 2
    CV_Chg = 3
    Rest = 4
    CCCV_Chg = 7

record_dtype = np.dtype([('Padding', 'i1'),
                         ('Record ID', 'i4'),
                         ('Cycle Index', 'i4'),
                         ('Step Index', 'i1'),
                         ('Step Type Code', 'i1'),
                         ('Time in Step', 'i4'),
                         ('Voltage', 'i4'),
                         ('Current', 'i4'),
                         ('Temperature', 'i8'),
                         ('Capacity', 'i8'),
                         ('Energy', 'i8'),
                         ('Timestamp', 'i8'),
                         ('idk', 'i4')
                         ])

step_names_mapping = np.array(['', 'CC Chg', 'CC DChg', 'CV Chg', 'Rest', '', '', 'CCCV Chg'])

def read_nda(file_path: str, drop_unknown_step=True):
    # TODO: grab information from headers
    f = open(file_path, 'rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    header_size = 2304
    second_header_pos = mm.find(b'\xff', header_size + 1)
    mm.seek(second_header_pos)
    data = np.frombuffer(mm.read(), record_dtype)
    df = pd.DataFrame(data)
    df['Voltage'] /= 10000 # V
    df['Current'] /= 10 # mA
    df['Capacity'] /= 36000 # mAh
    df['Energy'] /= 36000 # mWh
    df['Step Type'] = step_names_mapping[df['Step Type Code']]
    mm.close()
    f.close()
    df.drop(['Padding', 'idk'], inplace=True, axis=1)
    if drop_unknown_step:
        df = df[df['Step Type Code'] != 0]
    return df
