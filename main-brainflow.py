import time
import heartpy as hp
import matplotlib.pyplot as plt


from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter

import neurokit2 as nk




# https://brainflow.org/2022-12-03-brainflow-5-5-0/
# Sampling rate : https://github.com/EmotiBit/EmotiBit_Docs/blob/master/Working_with_emotibit_data.md#Data-type-sampling-rates
# pyPPG : https://pyppg.readthedocs.io/en/latest/tutorials/PPG_anal.html
# heartpy multiple channel : https://github.com/paulvangentcom/heartrate_analysis_python/issues/36
# neurokit : https://neuropsychology.github.io/NeuroKit/functions/ppg.html


sample_rate = 25.0


def main():
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    # params.ip_address = "225.1.1.1"
    board_id = BoardIds.EMOTIBIT_BOARD.value

    presets = BoardShim.get_board_presets(board_id)
    print (presets)
    
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    time.sleep(10)
    # ACC, GYR, MAGN
    #data_default = board.get_board_data(preset=BrainFlowPresets.DEFAULT_PRESET)
    # PPG 
    data_aux = board.get_board_data(preset=BrainFlowPresets.AUXILIARY_PRESET)
    # EDA, temperature
    data_anc = board.get_board_data(preset=BrainFlowPresets.ANCILLARY_PRESET)

    print("EDA", data_anc)
    print("PPG", len(data_aux), len(data_aux[0]), data_aux[1])
    board.stop_stream()
    board.release_session()
    #DataFilter.write_file(data_default, 'default.csv', 'w')
    DataFilter.write_file(data_aux, 'ppg.csv', 'w') # aux.csv
    DataFilter.write_file(data_anc, 'anc.csv', 'w')


    # cf. https://github.com/paulvangentcom/heartrate_analysis_python/blob/master/examples/1_regular_PPG/Analysing_a_PPG_signal.ipynb 
    ppg = data_aux[1] #hp.get_data('ppg.csv')

    signals, info = nk.ppg_process(ppg, sampling_rate=sample_rate)
    #ppg_elgendi = nk.ppg_clean(ppg, method='elgendi')

    nk.ppg_plot(signals, info)

    plt.show()


    ppg = data_aux[2] #hp.get_data('ppg.csv')

    signals, info = nk.ppg_process(ppg, sampling_rate=sample_rate)
    #ppg_elgendi = nk.ppg_clean(ppg, method='elgendi')

    nk.ppg_plot(signals, info)

    plt.show()

    ppg = data_aux[3] #hp.get_data('ppg.csv')

    signals, info = nk.ppg_process(ppg, sampling_rate=sample_rate)
    #ppg_elgendi = nk.ppg_clean(ppg, method='elgendi')

    nk.ppg_plot(signals, info)

    plt.show()

    """
    plt.figure(figsize=(12,4))
    plt.plot(ppg)
    plt.show()

    
    wd, m = hp.process(ppg, sample_rate = sample_rate)
    #set large figure
    plt.figure(figsize=(12,4))

    #call plotter
    hp.plotter(wd, m)

    #display measures computed
    for measure in m.keys():
        print('%s: %f' %(measure, m[measure]))
    """

    




if __name__ == "__main__":
    main()


