import os
import glob
import numpy as np
import pandas as pd
import yadg

def find_runs(x):
    """Find runs of consecutive items in an array."""
    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]
    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]
        # find run values
        run_values = x[loc_run_start]
        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths

def split_df(df, index_col):
    _, start, length = find_runs(df[index_col])
    return [df.iloc[s:s+l] for s, l in zip(start, length)]

class CellManager(object):
    def __init__(self, bclass, date, *addl_fields):
        self.prefix = f'{bclass}-{date}'
        if len(addl_fields) > 0:
            self.prefix += '-' + '-'.join(addl_fields)

    def __call__(self, batch_num):
        return self.prefix + '-' + str(batch_num).zfill(2)

    def make_id(self, num):
        return [self(i) for i in range(1, num + 1, 1)]

    def match_files(self, directory, suffix_pattern=''):
        # Some testing software uses `_` to add batch number, others use `-`
        full_pattern = self.prefix + '*' + suffix_pattern
        return glob.glob(os.path.join(directory, full_pattern))

    def load_nda_excel(self, directory, suffix_pattern=''):
        files = self.match_files(directory, suffix_pattern + '.xlsx')
        flist = sorted(files)
        dfs = [pd.read_excel(i, sheet_name=2) for i in flist]
        return dfs, flist

    def load_ec_eis(self, directory, suffix_pattern=''):
        files = self.match_files(directory, suffix_pattern + 'PEIS*.mpr')
        flist = sorted(files)
        data_list = list()
        for fp in flist:
            data = yadg.extractors.extract('eclab.mpr', fp)
            df = pd.DataFrame()
            df['Re(Z)'] = data['Re(Z)']
            df['-Im(Z)'] = data['-Im(Z)']
            data_list.append(df)
        return data_list, flist

def _get_capacity_units(df):
    headers = df.columns
    for h in headers:
        if h.startswith('Capacity'):
            return h
    else:
        raise ValueError('No capacity column')


class RateCapability(object):

    def __init__(self, dfs, cat_mass, c_rate, cell_id=None,
                 step_idx_col='Step Number', dchg_step='CC DChg'):
        self.n_data = len(dfs)
        self.cell_id = cell_id
        if len(cat_mass) != self.n_data:
            raise ValueError('Data length mismatch')
        self.cat_mass = cat_mass
        self.c_rate = c_rate
        self._dchg_step = dchg_step
        self._cap_col = _get_capacity_units(dfs[0])

        self.cycle_data = list()
        for df in dfs:
            self.cycle_data.append(split_df(df, step_idx_col))

    def discharge_profile(self, data_idx=0):
        profiles = list()
        for df in self.cycle_data[data_idx]:
            _step_name = df['Step Type'].iloc[0]
            if _step_name == self._dchg_step:
                df_sub = pd.DataFrame()
                df_sub['Voltage(V)'] = df['Voltage(V)']
                df_sub[self._cap_col] = df[self._cap_col] / self.cat_mass[data_idx]
                profiles.append(df_sub)
        return profiles

    def specific_capacity(self, data_idx=0):
        dchg_profile = self.discharge_profile(data_idx)
        caps = [df[self._cap_col].iloc[-1] for df in dchg_profile]
        ser = pd.Series(caps, index=self.c_rate)
        return ser

    def specific_capacity_all(self):
        df = pd.concat([self.specific_capacity(i) for i in range(self.n_data)], axis=1)
        if self.cell_id is not None:
            df.columns = self.cell_id
        return df


class Cycling(object):

    def __init__(self, df, step_idx_col='Step Number'):
        self.cycle_data = split_df(df, step_idx_col)
        self._cap_col = _get_capacity_units(df)
        self.chg_capacity = None
        self.dchg_capacity = None
        self._chg_dchg_capacity()

    def _chg_dchg_capacity(self):
        chg_cap = list()
        dchg_cap = list()
        _chg_this_cycle = 0
        _dchg_this_cycle = 0
        cycle_num = self.cycle_data[0]['Cycle Index'].iloc[0]
        for cy_df in self.cycle_data:
            print(cycle_num)
            _cycle_num = cy_df['Cycle Index'].iloc[0]
            if cycle_num != _cycle_num:
                chg_cap.append(_chg_this_cycle)
                dchg_cap.append(_dchg_this_cycle)
                _chg_this_cycle = 0
                _dchg_this_cycle = 0
                cycle_num = _cycle_num
            _step_name = cy_df['Step Type'].iloc[0]
            if _step_name.endswith(' DChg'):
                _dchg_this_cycle += cy_df[self._cap_col].iloc[-1]
            elif _step_name.endswith(' Chg'):
                _chg_this_cycle += cy_df[self._cap_col].iloc[-1]
        self.chg_capacity = np.array(chg_cap)
        self.dchg_capacity = np.array(dchg_cap)