import os
import numpy as np
import pandas as pd
import re
import time
import coloredPrinting as pr


"""
An object of this class gets generated as with two inputs: dataAdd and binSize. 
The dataAdd represents the output folder path for a Cooja scenario. All attributes 
and methods in use this address to read the out files of the Cooja scenario.
the binSize represents the tine interval in which we aim to get the mean and standard 
deviation of RPL control messages. 60 means 60 seconds.
"""



class MyDataSet():
    def __init__(self, dataAdd, binSize = 60):
        self.add = dataAdd
        # Cooja's time granularity is in microseconds (i.e., 1 min = 60000000 micro seconds)
        self.binSize = binSize * 1000000
        if self._check_address():
            self.moteAdd = self.add + "/mote-output.log"
            self.scriptAdd = self.add + "/script.log"
            self.eventsAdd = self.add + "/events.log"
            self.observations = self._observableToSink()
            self.steadyTime = int(self._readSteadyStateTime())
            self.attackTime = int(self._readAttackTime())
            self.attackTime = int((self.attackTime - self.steadyTime)/60000000)
            self._observableAfterSteady()
            self.noNodes = len(self.observations)
            self._save_obs_node()
            self.features = self._14_labeled_features()
            self._save_14_features()

    


    def _check_address(self):
        if (os.path.isdir(self.add) & os.path.isfile(self.add + "/mote-output.log")):
            # if(os.path.isfile(self.add + "/timeseries_60_sec_timestep.csv")):
            #     pr.prGreen("File already exists!")
            # else:
            print(" -----  Generating!")

            return True
        
        else:
            print("The folder or the mote-output.log file does not exist!")
            return False




    def _observableToSink(self):
        allLogs = self._allPackets()
        observations, counters = self._listOfDataFrames(logs = allLogs)

        #Here I have separated info for each node
        lookedfor = 0
        found = 0

        for i in range(allLogs.shape[0]):
            if (allLogs.loc[i,'mote'] == 1 and 'Received  message' in allLogs.loc[i,'message']):
                received_msg = allLogs.loc[i, 'message']
                received_request_num = int(received_msg.split('message')[1].split('(')[0].strip())

                # Extract sender from the 'from' field
                _, _, after_keyword = received_msg.partition('from')
                sender = int(after_keyword.split(':')[-1], 16)

                # Search backwards for the correct "Sending request <id>" by that sender
                lookedfor += 1
                j = i
                while j >= 0:
                    log_msg = allLogs.loc[j, 'message']
                    if ('Sending request' in log_msg and
                        allLogs.loc[j, 'mote'] == sender and
                        f'Sending request {received_request_num}' in log_msg):
                        found += 1
                        break
                    j -= 1

                # Only proceed if matching sender/request log is found
                if j >= 0:
                    msg_parts = allLogs.loc[j + 1, 'message'].split(',')
                    observations[str(sender)].loc[int(counters[str(sender)])] = [
                        allLogs.loc[j + 1, '# time'],
                        msg_parts[1].split(':')[1],
                        msg_parts[3].split(':')[1],
                        msg_parts[4].split(':')[1],
                        msg_parts[5].split(':')[1],
                        msg_parts[6].split(':')[1],
                        msg_parts[7].split(':')[1],
                        msg_parts[8].split(':')[1]
                    ]
                    counters[str(sender)] += 1
        for node in observations:
            observations[node] = observations[node].astype({'time':'float'})
            observations[node] = observations[node].astype({'rank':'float'})
            observations[node] = observations[node].astype({'disr':'float',})
            observations[node] = observations[node].astype({'diss':'float',})
            observations[node] = observations[node].astype({'dior':'float',})
            observations[node] = observations[node].astype({'dios':'float'})
            observations[node] = observations[node].astype({'diar':'float'})
            observations[node] = observations[node].astype({'tots':'float'})

        # Cleaning data 2
        for node in observations:
            l = observations[node].shape[0] - 1
            while(l != 0):
                myT = observations[node].loc[l,'time']
                myR = observations[node].loc[l,'rank']
                observations[node].loc[l,:] = observations[node].loc[l,:] - observations[node].loc[l-1,:]
                observations[node].loc[l,'time'] = myT
                observations[node].loc[l,'rank'] = myR
                l = l - 1
        
        print(lookedfor)
        print(found)
        return(observations)



    def _listOfDataFrames(self, logs):
        nodes = np.sort(logs['mote'].dropna().unique())
        nodes = nodes.astype(int)
        nodes = nodes.astype(str)
        noNodes = len(nodes)
        myDict1 = {}
        myDict2 = {}
        for i in range(1,noNodes,1):
            myDict1[nodes[i]] = self._emptyDataFrame()
            myDict2[nodes[i]] = int(0)
            
        return(myDict1,myDict2)


    def _allPackets(self):
        logs = pd.read_csv(self.moteAdd, sep = '\t')
        return(logs)

    def _emptyDataFrame(self):
        df = pd.DataFrame({'time': pd.Series(dtype = 'float'), 
                        'rank': pd.Series(dtype = 'float'), 
                        'disr': pd.Series(dtype = 'float'),
                        'diss': pd.Series(dtype = 'float'),
                        'dior': pd.Series(dtype = 'float'),
                        'dios': pd.Series(dtype = 'float'),
                        'diar': pd.Series(dtype = 'float'),
                        'tots': pd.Series(dtype = 'float')})
        return(df)

    def _readSteadyStateTime(self):
        f = open(self.scriptAdd)
        read = f.read()
        
        f.seek(0)
        
        arr = []
        
        line = 1
        for word in read:
            if word == '\n':
                line += 1

        for i in range(line):
            arr.append(f.readline())
        
        
        for i in range(line):
            if ('network steady state!' in arr[i] or 'Network steady state!' in arr[i]):
                break
        
        myString = arr[i]
        steadyTime = myString.split()[0]
    
        return(steadyTime)
    

    def _readStopTime(self):
        f = open(self.scriptAdd)
        read = f.read()
        
        f.seek(0)
        
        arr = []
        
        line = 1
        for word in read:
            if word == '\n':
                line += 1

        for i in range(line):
            arr.append(f.readline())	

        
        for i in range(line):
            if ('TEST OK' in arr[i]):
                break
        
        myString = arr[i]
        stopTime = myString.split()[0]
        
        return(stopTime)
    
    """
    def _readAttackTime(self):
        f = open(self.scriptAdd)
        read = f.read()
        
        f.seek(0)
        
        arr = []
        
        line = 1
        for word in read:
            if word == '\n':
                line += 1

        for i in range(line):
            arr.append(f.readline())
        
        
        for i in range(line):
            if ('network steady state!' in arr[i] or 'Network steady state!' in arr[i]):
                break
        
        myString = arr[i + 1]
        attackTime = myString.split()[0]
        
        return(attackTime)
    """


    def _readAttackTime(self):
        f = open(self.eventsAdd)
        read = f.read()
        
        f.seek(0)
        
        arr = []
        
        line = 1
        for word in read:
            if word == '\n':
                line += 1

        for i in range(line):
            arr.append(f.readline())
        
        
        for i in range(line):
            if ('network	steady-state' in arr[i]):
                break
        
        myString = arr[i + 1]
        attackTime = myString.split()[0]
        
        return(attackTime)



    def _observableAfterSteady(self):
        self.stopTime = int(self._readStopTime())
        self.stopTime = self.stopTime - self.steadyTime

        for node in self.observations:
            self.observations[node] = self.observations[node][(self.observations[node]['time'] > self.steadyTime)]
            self.observations[node]['time'] = self.observations[node]['time'] - self.steadyTime
            
            

    def _save_obs_node(self):
        for key, value in self.observations.items():
            value.to_csv(self.add + '/obs_' + str(key) + '.csv')    

        print("Observations Saved!")


    def _save_14_features(self):
        self.features.to_csv(self.add + '/features_timeseries_' + str(int(self.binSize/1000000)) + '_sec.csv')
        print("Labeled Timeseries Saved!")


    def _14_labeled_features(self):    
        t = 0
        _list_of_dataframes = []
        while (t <= self.stopTime):
            _data_frame = self._emptyDataFrame()
            for key, value in self.observations.items():
                _data_frame = pd.concat([_data_frame, value[value['time'].between(t, t + self.binSize)]], ignore_index=True)
            
            t = t + self.binSize
            _list_of_dataframes.append(_data_frame)
                
        # min max normalization should not be here!
        # _list_of_dataframes = [(df - df.min()) / (df.max() - df.min()) for df in _list_of_dataframes]
        # remove all NaN
        _list_of_dataframes = [df.fillna(0) for df in _list_of_dataframes]
        means_list = [df.mean() for df in _list_of_dataframes]
        std_list = [df.std() for df in _list_of_dataframes]
        # remove the last element of the series as they might be NA
        std_list = std_list[:-1]
        means_list = means_list[:-1]

        
        
        # concatenate 
        means_std_list = [pd.concat([s1, s2]) for s1, s2 in zip(means_list, std_list)]
        df = pd.concat([s.to_frame().T for s in means_std_list], ignore_index=True)
        
        # Labeling

        label = np.ones(shape = df.shape[0])
    
        for i in range(int(self.attackTime - 1)):
            label[i] = 0
        
    
        df['label'] = label.tolist()
    
        # remove time from each series
        df = df.drop('time', axis = 1)
        #df = df.drop('City', axis=1)

        #std_list = [s.iloc[1:] for s in std_list]
        #means_list = [s.iloc[1:] for s in means_list]

        return df
        



        



