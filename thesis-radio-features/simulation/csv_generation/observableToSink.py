import os
import re
import numpy as np
import pandas as pd


class MyDataSet():
    def __init__(self, dataAdd, binSize=60):
        self.add = dataAdd
        self.binSize = binSize * 1000000
        if self._check_address():
            self.moteAdd = self.add + "/mote-output.log"
            self.scriptAdd = self.add + "/script.log"
            self.eventsAdd = self.add + "/events.log"
            self.observations = self._observableToSink()
            self.steadyTime = int(self._readSteadyStateTime())
            self.attackTime = int(self._readAttackTime())
            self.attackTime = int((self.attackTime - self.steadyTime) / 60000000)
            self._observableAfterSteady()
            self.noNodes = len(self.observations)
            self._save_obs_node()
            self.features = self._22_labeled_features()
            self._save_22_features()

    def _check_address(self):
        if os.path.isdir(self.add) and os.path.isfile(self.add + "/mote-output.log"):
            print(" -----  Generating!")
            return True
        return False

    def _observableToSink(self):
        allLogs = pd.read_csv(
            self.moteAdd,
            sep='\t',
            dtype={'mote': float, 'message': str}
        )

        nodes = np.sort(allLogs['mote'].dropna().unique()).astype(int).astype(str)
        observations = {node: self._emptyDataFrame() for node in nodes if node != '1'}
        counters = {node: 0 for node in nodes if node != '1'}

        energy_data = {
            node: {
                'cpu': 0.0,
                'lpm': 0.0,
                'deeplpm': 0.0,
                'tx': 0.0,
                'rx': 0.0
            }
            for node in nodes if node != '1'
        }

        for i in range(allLogs.shape[0]):
            if pd.isna(allLogs.loc[i, 'mote']):
                continue

            msg = str(allLogs.loc[i, 'message']).strip()
            mote_id = str(int(allLogs.loc[i, 'mote']))

            if mote_id in energy_data:
                m_energy = re.search(
                    r'CPU\s+(\d+)s\s+LPM\s+(\d+)s\s+DEEP\s+LPM\s+(\d+)s',
                    msg
                )
                if m_energy:
                    energy_data[mote_id]['cpu'] = float(m_energy.group(1))
                    energy_data[mote_id]['lpm'] = float(m_energy.group(2))
                    energy_data[mote_id]['deeplpm'] = float(m_energy.group(3))

                m_radio = re.search(
                    r'Radio\s+LISTEN\s+(\d+)s\s+TRANSMIT\s+(\d+)s',
                    msg
                )
                if m_radio:
                    energy_data[mote_id]['rx'] = float(m_radio.group(1))
                    energy_data[mote_id]['tx'] = float(m_radio.group(2))

            if mote_id == '1' and 'Received' in msg and 'message' in msg:
                try:
                    num = int(re.findall(r'message\s+(\d+)', msg)[0])
                    sender = str(int(msg.partition('from')[2].split(':')[-1].strip(), 16))

                    rssi_match = re.search(r'rssi\s+(-?\d+)', msg)
                    rssi_value = float(rssi_match.group(1)) if rssi_match else np.nan
                    if not np.isnan(rssi_value) and rssi_value > 32767:
                        rssi_value = rssi_value - 65536

                    lqi_match = re.search(r'lqi\s+(\d+)', msg)
                    lqi_value = float(lqi_match.group(1)) if lqi_match else 0.0
                except Exception:
                    continue

                if sender not in observations:
                    continue

                for j in range(i - 1, -1, -1):
                    prev_msg = str(allLogs.loc[j, 'message'])

                    if (
                        'Sending request' in prev_msg and
                        not pd.isna(allLogs.loc[j, 'mote']) and
                        str(int(allLogs.loc[j, 'mote'])) == sender and
                        f'request {num}' in prev_msg
                    ):
                        data_row = j + 1
                        if data_row >= allLogs.shape[0]:
                            break

                        data_msg = str(allLogs.loc[data_row, 'message'])
                        if 'DATA:' in data_msg:
                            d_nums = re.findall(r'\d+', data_msg)
                            if len(d_nums) >= 9:
                                observations[sender].loc[counters[sender]] = [
                                    float(allLogs.loc[data_row, '# time']),
                                    float(d_nums[1]),
                                    float(d_nums[3]),
                                    float(d_nums[4]),
                                    float(d_nums[5]),
                                    float(d_nums[6]),
                                    float(d_nums[7]),
                                    float(d_nums[8]),
                                    energy_data[sender]['cpu'],
                                    energy_data[sender]['lpm'],
                                    energy_data[sender]['deeplpm'],
                                    energy_data[sender]['tx'],
                                    energy_data[sender]['rx'],
                                    rssi_value,
                                    lqi_value
                                ]
                                counters[sender] += 1
                        break

        return observations

    def _emptyDataFrame(self):
        cols = [
            'time', 'rank', 'disr', 'diss', 'dior', 'dios', 'diar', 'tots',
            'cpu', 'lpm', 'deeplpm', 'tx', 'rx', 'rssi', 'lqi'
        ]
        return pd.DataFrame({c: pd.Series(dtype='float') for c in cols})

    def _readSteadyStateTime(self):
        with open(self.scriptAdd) as f:
            for line in f:
                if 'network steady state!' in line or 'Network steady state!' in line:
                    return line.split()[0]
        return 0

    def _readAttackTime(self):
        with open(self.eventsAdd) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if 'network\tsteady-state' in line and i + 1 < len(lines):
                    return lines[i + 1].split()[0]
        return 0

    def _readStopTime(self):
        with open(self.scriptAdd) as f:
            for line in f:
                if 'TEST OK' in line:
                    return line.split()[0]
        return 0

    def _observableAfterSteady(self):
        self.steadyTime = int(self.steadyTime)
        self.stopTime = int(self._readStopTime()) - self.steadyTime

        for node in self.observations:
            df = self.observations[node]
            df['time'] = pd.to_numeric(df['time'], errors='coerce')
            df = df[df['time'] > self.steadyTime].copy()
            df['time'] -= self.steadyTime
            self.observations[node] = df

    def _save_obs_node(self):
        for k, v in self.observations.items():
            v.to_csv(f"{self.add}/obs_{k}.csv")

    def _22_labeled_features(self):
        base_cols = [
            'rank', 'disr', 'diss', 'dior', 'dios', 'diar', 'tots',
            'cpu', 'lpm', 'deeplpm', 'tx', 'rx', 'rssi', 'lqi'
        ]

        if not self.observations or all(v.empty for v in self.observations.values()):
            print("⚠ No parsed observations, writing empty features file")
            return pd.DataFrame(columns=base_cols + base_cols + ['label'])

        t = 0
        dfs = []

        while t <= self.stopTime:
            current_obs = []
            for v in self.observations.values():
                filtered = v[v['time'].between(t, t + self.binSize)]
                if not filtered.empty:
                    current_obs.append(filtered)

            if current_obs:
                combined = pd.concat(current_obs, ignore_index=True)
                dfs.append(combined.apply(pd.to_numeric, errors='coerce').fillna(0))
            else:
                dfs.append(self._emptyDataFrame())

            t += self.binSize

        if len(dfs) <= 1:
            print("⚠ No usable windows, writing empty features file")
            return pd.DataFrame(columns=base_cols + base_cols + ['label'])

        means = [df.mean(numeric_only=True) for df in dfs[:-1]]
        stds = [df.std(numeric_only=True) for df in dfs[:-1]]

        res = [pd.concat([m, s]) for m, s in zip(means, stds) if not m.empty or not s.empty]

        if not res:
            print("⚠ No objects to concatenate, writing empty features file")
            return pd.DataFrame(columns=base_cols + base_cols + ['label'])

        df_final = pd.concat([s.to_frame().T for s in res], ignore_index=True).fillna(0)

        label = np.ones(df_final.shape[0])
        limit = min(len(label), int(self.attackTime))
        if limit > 0:
            label[:limit - 1] = 0
        df_final['label'] = label

        return df_final.drop('time', axis=1, errors='ignore')

    def _save_22_features(self):
        self.features.to_csv(f"{self.add}/features_energy_60_sec.csv")
        print("Done: 22 Features Saved!")
