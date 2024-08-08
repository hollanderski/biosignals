import time
import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter

import neurokit2 as nk

measures = {}

sample_rate = 25.0

def main():
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    board_id = BoardIds.EMOTIBIT_BOARD.value

    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    try:
        while True:
            time.sleep(10)  # Collect data every 5 second


            # PPG processing 
            data_aux = board.get_board_data(preset=BrainFlowPresets.AUXILIARY_PRESET)

            if data_aux.size > 0:
                ppg = data_aux[3]  # Extract PPG data from the relevant channel
                signals, info = nk.ppg_process(ppg, sampling_rate=sample_rate)
                bpm = signals['PPG_Rate'].mean()  # Extract the BPM value   /!\ should handle Nan             
                print(f"BPM: {bpm:.2f}")  # Print the BPM value
                measures["bpm"] = bpm
            else:
                print("No PPG data collected.")


            # EDA, temperature : package_num_channel    eda_channels    temperature_channels    other_channels  timestamp_channel   marker_channel
            data_anc = board.get_board_data(preset=BrainFlowPresets.ANCILLARY_PRESET)
            if data_anc.size > 0:
                eda_signal = data_anc[1]  # Extract PPG data from the relevant channel

                print("EDA signal ", eda_signal)
                eda_output = nk.eda_process(eda_signal, sampling_rate=sample_rate)
                vals_eda = eda_output[0]["EDA_Clean"]
                
                psd = nk.signal_psd(vals_eda, method="fft", min_frequency=0, max_frequency=2.4, show=False)

                print("EDA psd ", psd)
                
                measures['RiseTime'] = np.mean(eda_output[0]['SCR_RiseTime'])                
                measures['EDA_Tonic'] = np.mean(eda_output[0]['EDA_Tonic'])
                measures['EDA_Mean'] = np.mean(vals_eda)   
                measures["gmrs"] = np.sqrt(np.sum(psd["Power"].values**2))
                print("EDA features ", measures)
                
            else:
                print("No EDA data collected.")


    except KeyboardInterrupt:
        print("Data collection stopped.")
    finally:
        board.stop_stream()
        board.release_session()

if __name__ == "__main__":
    main()



'''

EDA processing 

 eda_output = nk.eda_process(eda_signal, sampling_rate=1000)
        vals_eda = eda_output[0]["EDA_Clean"]
        
        psd = nk.signal_psd(vals_eda, method="fft", min_frequency=0, max_frequency=2.4, show=False)
        
        measures['RiseTime'] = np.mean(eda_output[0]['SCR_RiseTime'])                
        measures['EDA_Tonic'] = np.mean(eda_output[0]['EDA_Tonic'])
        measures['EDA_Mean'] = np.mean(vals_eda)   
        measures["gmrs"] = np.sqrt(np.sum(psd["Power"].values**2))


'''