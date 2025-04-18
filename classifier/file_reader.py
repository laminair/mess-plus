import pandas as pd
import glob


def read_files_from_folder(folder_path: str, file_ext: str = ".csv"):
    file_list = glob.glob(f"{folder_path}/*{file_ext}")

    # Use outer join to include all columns from all files
    combined_df = pd.concat(
        [pd.read_csv(file) for file in file_list],
        ignore_index=True,
        join="outer"
    )  # 'outer' keeps all columns

    return combined_df