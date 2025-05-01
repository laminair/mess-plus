import pandas as pd
import glob


def read_files_from_folder(folder_path: str, file_ext: str = ".csv"):
    file_list = glob.glob(f"{folder_path}/*{file_ext}")
    print(len(file_list))

    combined_df = pd.concat(
        [pd.read_csv(file, index_col="doc_id") for file in file_list],
    ).drop_duplicates()

    return combined_df