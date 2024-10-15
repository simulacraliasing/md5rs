import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def merge_frames(df: pd.DataFrame):
    df = df.filter(items=["folder_id", "file_id", "file_path", "shoot_time", "label"])
    df = df.dropna(subset=["label"])
    df["shoot_time"] = pd.to_datetime(df["shoot_time"]).dt.tz_localize(None)
    priority_order = {"Animal": 1, "Person": 2, "Vehicle": 3, "Blank": 4}

    # Function to get the most prioritized label
    def get_priority_label(group):
        return group.loc[group["label"].map(priority_order).idxmin()]

    # Aggregate the DataFrame
    result = df.groupby("file_id").apply(get_priority_label).reset_index(drop=True)

    return result


def date_parser(s):
    return pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S")


class Cluster:
    def __init__(self, target: Path) -> None:
        self.target = target

    @staticmethod
    def create_folders(folder_path: Path) -> None:
        animal_folder = folder_path / "Animal"
        person_folder = folder_path / "Person"
        vehicle_folder = folder_path / "Vehicle"
        blank_folder = folder_path / "Blank"
        for folder in [animal_folder, person_folder, vehicle_folder, blank_folder]:
            folder.mkdir(exist_ok=True)

    @staticmethod
    def is_video_time_end_time(folder_df: pd.DataFrame) -> bool:
        seq = []
        seq_size = []
        for i, file in enumerate(folder_df.itertuples()):
            if len(seq) == 0 or abs(
                file.shoot_time - seq[-1].shoot_time
            ) < pd.Timedelta(seconds=5):
                seq.append(file)
            else:
                seq_size.append(len(seq))
                seq = [file]
        cross_diff = np.array(seq_size[1:]) - np.array(seq_size[:-1])
        cross_diff = abs(cross_diff)
        cross_diff_mean = np.mean(cross_diff)
        return cross_diff_mean > 0.5

    def parse_csv(self, csv: Path) -> pd.DataFrame:
        return pd.read_csv(
            csv,
            encoding="utf-8",
            usecols=range(9),
        )

    def parse_json(self, json_path: Path) -> pd.DataFrame:
        df = pd.read_json(json_path, orient="records", encoding="utf-8")
        print(df.head())
        return df

    def move_seq(
        self,
        seq: list[pd.Series],
        folder_path: Path,
        folder_df: pd.DataFrame,
    ) -> pd.DataFrame:
        animal_flag = False
        person_flag = False
        vehicle_flag = False
        for file in seq:
            if file.label == "Animal":
                animal_flag = True
                break
            elif file.label == "Person":
                person_flag = True
            elif file.label == "Vehicle":
                vehicle_flag = True
        if animal_flag:
            label = "Animal"
        elif person_flag:
            label = "Person"
        elif vehicle_flag:
            label = "Vehicle"
        else:
            label = "Blank"
        for file in seq:
            self.df.loc[self.df["file_id"] == file.file_id, "seq_id"] = self.seq_id
            self.df.loc[self.df["file_id"] == file.file_id, "seq_label"] = label
            dest = folder_path / label
            logger.info(f"Moving {file.file_path} to {str(dest)}")
            try:
                shutil.move(file.file_path, dest)
                self.df.loc[self.df["file_id"] == file.file_id, "moved"] = True
            except Exception as e:
                logger.error(f"Failed to move file: {e}")
                self.df.loc[self.df["file_id"] == file.file_id, "moved"] = False
            folder_df = folder_df[folder_df["file_id"] != file.file_id]
        return folder_df

    def organize(
        self,
        result: Path,
        guess: bool = False,
    ):

        self.start = time.perf_counter()
        if result.suffix == ".json":
            self.df = self.parse_json(result)
        else:
            self.df = self.parse_csv(result)
        self.df = merge_frames(self.df)
        self.df = self.df[self.df["label"].notnull()]
        self.df["seq_id"] = np.nan
        self.df["seq_label"] = ""
        self.df["moved"] = False
        self.total_file_count = len(self.df.index)
        folders = self.df["folder_id"].unique()
        self.seq_id = 0
        for folder in folders:
            folder_df = self.df[self.df["folder_id"] == folder]
            folder_path = Path(folder_df.iloc[0]["file_path"]).parent
            logger.info(f"Processing folder {folder_path}")
            self.create_folders(folder_path)
            folder_df = folder_df.sort_values(by=["shoot_time", "file_id"]).reset_index(
                drop=True
            )
            diff = folder_df.index - folder_df["file_id"].astype(int)
            std = np.std(diff)
            is_right_seq = std < 1
            folder_df["shoot_time"] = pd.to_datetime(
                folder_df["shoot_time"], errors="coerce"
            )
            if folder_df["shoot_time"].isnull().values.any():
                logger.warning("Shoot time is null")
                is_right_seq = False
            if is_right_seq and not self.is_video_time_end_time(folder_df):
                logger.info(f"Folder {folder_path}: Time model")
                # 时间顺序正确 按照时间顺序分包
                seq = []
                for i, file in enumerate(folder_df.itertuples()):
                    if len(seq) == 0 or abs(
                        file.shoot_time - seq[-1].shoot_time
                    ) < pd.Timedelta(seconds=5):
                        seq.append(file)
                    else:
                        self.seq_id += 1
                        folder_df = self.move_seq(seq, folder_path, folder_df)
                        seq = [file]
                if len(seq) > 0:
                    folder_df = self.move_seq(seq, folder_path, folder_df)
            elif is_right_seq and self.is_video_time_end_time(folder_df) and guess:
                logger.info(f"Processing folder {folder_path}: Fallback to Guess model")
                folder_df = self.guess_model(folder_df, folder_path)
            elif is_right_seq and self.is_video_time_end_time(folder_df) and not guess:
                logger.info(f"Processing folder {folder_path}: Fallback to No Guess model")
                folder_df = self.non_guess_model(folder_df, folder_path)
            elif not is_right_seq and guess:
                logger.info(f"Processing folder {folder_path}: Guess model")
                folder_df = self.guess_model(folder_df, folder_path)
                # 时间顺序不正确 按照文件名顺序分包

            elif not is_right_seq and not guess:
                logger.info(f"Processing folder {folder_path}: No Guess Model")
                # 时间顺序不正确 不适用猜测模式
                folder_df = self.non_guess_model(folder_df, folder_path)
        self.df = self.df.filter(items=["file_id", "seq_id", "seq_label", "moved"])
        orgnized_output = os.path.splitext(result)[0] + "_organized.csv"
        self.df.to_csv(orgnized_output, encoding="utf-8-sig", index=False)

    def guess_model(
        self,
        folder_df: pd.DataFrame,
        folder_path: Path,
    ) -> pd.DataFrame:
        while len(folder_df.index) > 0:
            folder_df = folder_df.sort_values(by=["file_id"]).reset_index(drop=True)
            file_paths = folder_df["file_path"].tolist()
            first_90_path = file_paths[:90]
            first_90_suffix = [Path(file).suffix.lower() for file in first_90_path]
            window_size = 1
            # 以window_size相同的步长滑动window读取first_90_path
            # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
            # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
            max_count = 0
            best_window_size = 0
            for window_size in range(1, 6):
                fold = len(first_90_suffix) // window_size
                right_count = 0
                stack = []
                for i in range(0, window_size * (fold - 1), window_size):
                    window = first_90_suffix[i : i + window_size]
                    if len(stack) == 0:
                        stack = window
                    elif window == stack:
                        right_count += 1
                    else:
                        right_count -= 1
                        stack = window
                if right_count > max_count:
                    max_count = right_count
                    best_window_size = window_size
                if window_size >= 5 and best_window_size == 0:
                    folder_df = self.non_guess_model(folder_df, folder_path)
            logger.info(f"Best window size: {best_window_size}")
            offset = 0
            while (
                first_90_suffix[offset : offset + best_window_size]
                != first_90_suffix[
                    offset + best_window_size : offset + 2 * best_window_size
                ]
            ):
                offset += 1
            logger.info(f"Offset: {offset}")
            for file in folder_df.iloc[:offset].itertuples():
                self.seq_id += 1
                self.df.loc[self.df["file_id"] == file.file_id, "seq_id"] = self.seq_id
                self.df.loc[self.df["file_id"] == file.file_id, "seq_label"] = (
                    file.label
                )
                dest = folder_path / file.label
                logger.info(f"Moving {file.file_path} to {str(dest)}")
                try:
                    shutil.move(file.file_path, dest)
                    self.df.loc[self.df["file_id"] == file.file_id, "moved"] = True
                except Exception as e:
                    logger.error(f"Failed to move file: {e}")
                    self.df.loc[self.df["file_id"] == file.file_id, "moved"] = False
                folder_df = folder_df[folder_df["file_id"] != file.file_id]
            seq = []
            suffix_stack = []
            for i, file in enumerate(folder_df.itertuples()):
                if i == 0:
                    seq.append(file)
                else:
                    if i % best_window_size != 0:
                        seq.append(file)
                    else:
                        if len(suffix_stack) == 0:
                            for _file in seq:
                                suffix_stack.append(Path(_file.file_path).suffix)
                        else:
                            new_suffix_stack = [
                                Path(_file.file_path).suffix for _file in seq
                            ]
                            if new_suffix_stack != suffix_stack:
                                break
                        self.seq_id += 1
                        folder_df = self.move_seq(seq, folder_path, folder_df)
                        seq = [file]
        return folder_df

    def non_guess_model(
        self,
        folder_df: pd.DataFrame,
        folder_path: Path,
    ) -> pd.DataFrame:
        for _file in folder_df.itertuples():
            dest = folder_path / _file.label
            self.seq_id += 1
            self.df.loc[self.df["file_id"] == _file.file_id, "seq_id"] = self.seq_id
            self.df.loc[self.df["file_id"] == _file.file_id, "seq_label"] = _file.label
            logger.info(f"Moveing {_file.file_path} to {str(dest)}")
            try:
                shutil.move(_file.file_path, dest)
                self.df.loc[self.df["file_id"] == _file.file_id, "moved"] = True
            except Exception as e:
                logger.error(f"Failed to move file: {e}")
                self.df.loc[self.df["file_id"] == _file.file_id, "moved"] = False
            folder_df = folder_df[folder_df["file_id"] != _file.file_id]
        return folder_df

    def redo_orgnize(self):
        self.total_file_count = 0
        for root, _, files in os.walk(self.target):
            for file in files:
                file_path = Path(root) / file
                if str(file_path.parts[-2]) in ["Animal", "Person", "Vehicle", "Blank"]:
                    dest_path = Path(root).parent / file
                    try:
                        shutil.move(file_path, dest_path)
                        logger.info(f"Moving {file_path} to {dest_path}")
                    except Exception as e:
                        logger.error(f"Failed to move file: {e}")


def main(result: Path, redo: bool = False, guess: bool = False):
    cluster = Cluster(result.parent)
    if redo:
        cluster.redo_orgnize()
    else:
        cluster.organize(result, guess=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize files by sequence")
    parser.add_argument(
        "--result",
        type=str,
        help="Path to the result file",
    )
    parser.add_argument(
        "--guess",
        action="store_true",
        help="Use guess model",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Redo organize",
    )

    args = parser.parse_args()

    if not args.result:
        logger.error("Please provide the path to the result file")
        sys.exit(1)

    result = Path(args.result)

    result = result.absolute()

    extension = result.suffix.lower()

    if extension not in [".csv", ".json"]:
        logger.error("Invalid file extension")
        sys.exit(1)

    main(result, guess=args.guess, redo=args.redo)
