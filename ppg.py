import time
import heartpy as hp
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter

import neurokit2 as nk

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
            time.sleep(5)  # Collect data every 10 second
            data_aux = board.get_board_data(preset=BrainFlowPresets.AUXILIARY_PRESET)


            if data_aux.size > 0:
                ppg = data_aux[3]  # Extract PPG data from the relevant channel
                signals, info = nk.ppg_process(ppg, sampling_rate=sample_rate)
                bpm = signals['PPG_Rate'].mean()  # Extract the BPM value   /!\ should handle Nan             
                print(f"BPM: {bpm:.2f}")  # Print the BPM value
            else:
                print("No data collected.")
    except KeyboardInterrupt:
        print("Data collection stopped.")
    finally:
        board.stop_stream()
        board.release_session()

if __name__ == "__main__":
    main()
