import atexit
import csv
import time

import brainflow.exit_codes
import yaml
from brainflow import BrainFlowError, DataFilter
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from pathlib import Path
from pprint import pprint

from utils import load_config, create_folder_structure


class GSR:
    def __init__(self, config: yaml, root_data_path: Path, board_id: BoardIds = BoardIds.EMOTIBIT_BOARD):
        self.board = None
        self.board_id = board_id
        self.config = config
        self.file_movement = Path(root_data_path / 'gsr' / config["DATA_CAPTURE"]["GSR"]["MOVEMENT"])
        self.file_ppg = Path(root_data_path / 'gsr' / config["DATA_CAPTURE"]["GSR"]["PPG"])
        self.file_eda = Path(root_data_path / 'gsr' / config["DATA_CAPTURE"]["GSR"]["EDA"])
        print(self)
        print("EDA FILE", self.file_eda)
        BoardShim.enable_dev_board_logger()

        params = BrainFlowInputParams()
        self.board = BoardShim(BoardIds.EMOTIBIT_BOARD, params)
        self.board.prepare_session()
        pprint(BoardShim.get_board_descr(board_id, preset=BrainFlowPresets.ANCILLARY_PRESET))
        pprint(self.board.get_other_channels(board_id=self.board.board_id,
                                             preset=BrainFlowPresets.ANCILLARY_PRESET))

    def prep_stream_file(self):
        headers = [
            ["package_num_channel", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "mag_x", "mag_y", "mag_z",
             "timestamp_channel", "marker_channel"]
            , ["package_num_channel", "PPG_1", "PPG_2", "PPG_3", "timestamp_channel", "marker_channel"],
            ["package_num_channel", "eda_channels", "temperature_channels", "other_channels",
             "timestamp_channel", "marker_channel"]]
        file_list = [self.file_movement, self.file_ppg, self.file_eda]

        comb = zip(headers, file_list)
        for header, file in comb:
            print(header, file)
            try:
                with open(file, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter="\t")
                    csvwriter.writerow(header)
            except Exception as e:
                print(e)

    def stream_to_file(self):
        self.board.add_streamer(streamer_params="file://{}:a".format(self.file_eda),
                                preset=BrainFlowPresets.ANCILLARY_PRESET)

        self.board.add_streamer(streamer_params="file://{}:a".format(self.file_movement),
                                preset=BrainFlowPresets.DEFAULT_PRESET)

        self.board.add_streamer(streamer_params="file://{}:a".format(self.file_ppg),
                                preset=BrainFlowPresets.AUXILIARY_PRESET)

    def launch_gsr(self):
        self.prep_stream_file()
        self.stream_to_file()
        self.board.start_stream()
        print("GSR started")

    # @atexit.register
    def stop_gsr(self):
        # make sure we clean the connection if the application is stopped
        print("finishing up GSR")
        # self.stop_sd_recording()
        self.board.stop_stream()
        self.board.release_session()


if __name__ == '__main__':
    config = load_config(Path(Path.cwd().parent / "conf.yaml"))

    gsr = GSR(
        config=config,
        root_data_path=Path.cwd(),
    )
    gsr.launch_gsr()

    time.sleep(5)
    gsr.stop_gsr()
