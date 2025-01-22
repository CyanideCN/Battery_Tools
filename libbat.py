import os
import glob

import numpy as np
import pandas as pd
import yadg
import NewareNDA

from ndav8 import read_nda, NDAStepType

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

def split_df(df: pd.DataFrame, index_col: str) -> list[pd.DataFrame]:
    _, start, length = find_runs(df[index_col])
    return [df.iloc[s:s+l] for s, l in zip(start, length)]

class CellManager(object):
    def __init__(self, bclass: str, date: str, *addl_fields):
        self.prefix = f'{bclass}-{date}'
        if len(addl_fields) > 0:
            self.prefix += '-' + '-'.join(addl_fields)

    def __call__(self, batch_num: int):
        return self.prefix + '-' + str(batch_num).zfill(2)

    def make_id(self, num: int):
        return [self(i) for i in range(1, num + 1, 1)]

    def match_files(self, directory, suffix_pattern='', exclude=''):
        # Some testing software uses `_` to add batch number, others use `-`
        full_pattern = self.prefix + '*' + suffix_pattern
        result = glob.glob(os.path.join(directory, full_pattern))
        if exclude:
            result = [i for i in result if exclude not in i]
        return result

    def load_nda(self, directory, suffix_pattern='', exclude=''):
        files = self.match_files(directory, suffix_pattern + '.nda', exclude=exclude)
        flist = sorted(files)
        rename_dict = {'Voltage': 'Voltage(V)'}
        dfs = [read_nda(i).rename(columns=rename_dict) for i in flist]
        cids = [self._parse_cell_id(i) for i in flist]
        return dfs, cids

    def load_nda_excel(self, directory, suffix_pattern='', exclude=''):
        files = self.match_files(directory, suffix_pattern + '.xlsx', exclude=exclude)
        flist = sorted(files)
        dfs = [pd.read_excel(i, sheet_name=2, engine='calamine') for i in flist]
        cids = [self._parse_cell_id(i) for i in flist]
        return dfs, cids

    def load_ndax(self, directory, suffix_pattern='', exclude=''):
        files = self.match_files(directory, suffix_pattern + '.ndax', exclude=exclude)
        flist = sorted(files)
        dfs = [NewareNDA.read(i) for i in flist]
        cids = [self._parse_cell_id(i) for i in flist]
        return dfs, cids

    def load_ec_eis(self, directory, suffix_pattern='', exclude=''):
        files = self.match_files(directory, suffix_pattern + 'PEIS*.mpr', exclude=exclude)
        flist = sorted(files)
        data_list = list()
        for fp in flist:
            data = yadg.extractors.extract('eclab.mpr', fp)
            df = pd.DataFrame()
            df['freq'] = data['freq']
            df['Re(Z)'] = data['Re(Z)']
            df['-Im(Z)'] = data['-Im(Z)']
            data_list.append(df)
        cids = [self._parse_cell_id(i) for i in flist]
        return data_list, cids

    def _parse_cell_id(self, raw_str: str):
        fname = os.path.split(raw_str)[1]
        if fname.startswith(self.prefix):
            fn_strip = fname.replace(self.prefix, '', 1)
            if fn_strip[0] in ['-', '_']:
                batch_num = ''
                for chars in fn_strip[1:]:
                    if chars.isdigit():
                        batch_num += chars
                    else:
                        break
                return f'{self.prefix}-{batch_num}'
        return None

class CellMetadata(object):

    def __init__(self, excel_path):
        self.data = pd.read_excel(excel_path, index_col=0)

    def get(self, cell_id, key):
        return self.data[key][cell_id]

def _get_capacity_units(df: pd.DataFrame) -> str:
    headers = df.columns
    for h in headers:
        if h.startswith('Capacity'):
            return h
    else:
        raise ValueError('No capacity column')


class RateCapability(object):

    def __init__(self, dfs, c_rate=None, active_mass=None, cell_id=None, metadata=None,
                 step_idx_col='Step Number', dchg_step='CC DChg', voltage_window=(2.5, 4.2),
                 exclude_partial_cyc=True, norm_target='anode'):
        self.n_data = len(dfs)
        self.cell_id = cell_id
        if active_mass is None:
            if cell_id and metadata:
                match norm_target:
                    case 'anode':
                        self.active_mass = metadata.get(cell_id, 'Anode Active Mass').values
                    case 'cathode':
                        self.active_mass = metadata.get(cell_id, 'Cathode Active Mass').values
            else:
                raise ValueError('Active mass is missing')
        else:
            self.active_mass = active_mass
        if len(self.active_mass) != self.n_data:
            raise ValueError('Data length mismatch')
        self.c_rate = c_rate
        self._dchg_step = dchg_step
        self._cap_col = _get_capacity_units(dfs[0])

        self.cycle_data: list[list[pd.DataFrame]] = list()
        for df in dfs:
            self.cycle_data.append(split_df(df, step_idx_col))

        self._SPEC_CAP_NAME = 'Specific_Capacity'

        # TODO: Generate a list of step type for each step

    def _quality_check(self, voltage_window, tol=0.05):
        v_low, v_high = voltage_window
        cycle_data = list()
        for df_segs in self.cycle_data:
            tmp_df_list = list()
            for df in df_segs:
                # Only remove incomplete discharge steps as for now
                _step_name = df['Step Type'].iloc[0]
                if _step_name == self._dchg_step:
                    _vmin = df['Voltage(V)'].iloc[-1]
                    if (_vmin - v_low) > tol:
                        continue
                    else:
                        tmp_df_list.append(df)
                else:
                    tmp_df_list.append(df)
            cycle_data.append(tmp_df_list)
        return cycle_data

    def discharge_profile(self, data_idx=0):
        profiles: list[pd.DataFrame] = list()
        for df in self.cycle_data[data_idx]:
            _step_name = df['Step Type'].iloc[0]
            if _step_name == self._dchg_step:
                df_sub = pd.DataFrame()
                df_sub['Voltage'] = df['Voltage(V)']
                df_sub[self._SPEC_CAP_NAME] = df[self._cap_col] / self.active_mass[data_idx]
                profiles.append(df_sub)
        return profiles

    def specific_capacity(self, data_idx=0):
        if not self.c_rate:
            raise ValueError('C-rate required.')
        dchg_profile = self.discharge_profile(data_idx)
        caps = [df[self._SPEC_CAP_NAME].iloc[-1] for df in dchg_profile]
        ser = pd.Series(caps, index=self.c_rate)
        return ser

    def specific_capacity_all(self):
        df = pd.concat([self.specific_capacity(i) for i in range(self.n_data)], axis=1)
        if self.cell_id is not None:
            df.columns = self.cell_id
        return df

    def discharge_current(self, data_idx=0):
        current = list()
        for df in self.cycle_data[data_idx]:
            _step_name = df['Step Type'].iloc[0]
            if _step_name == self._dchg_step:
                i_row = df['Current(mA)']
                current.append(i_row.iloc[-2])
        return np.array(current)


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

    def capacity_retention(self, skip=1):
        return self.dchg_capacity[skip:] / self.dchg_capacity[skip]

    def coulombic_eff(self, skip=1):
        return self.dchg_capacity[skip:] / self.chg_capacity[skip:]
    
def dcir(data, take_idx=9):
    _, start, length = find_runs(data['Step Type Code'] == NDAStepType.Rest)
    rest_seg = data.iloc[:length[0]]
    dchg_seg = data.iloc[start[1]:start[1] + length[1]]
    v_dchg = dchg_seg.iloc[take_idx]['Voltage']
    v_rest = rest_seg['Voltage'].iloc[-10:].mean() # Last 10s avg
    ir = (v_dchg - v_rest) / (dchg_seg.iloc[0]['Current'] / 1000)
    return ir