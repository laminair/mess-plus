import pandas as pd
import os

from datetime import datetime
from lm_eval.api.task import Instance

from typing import Dict


class StreamingDataProcessor:
    def __init__(self, save_path='data/', file_prefix='streaming_data_', save_frequency=100):
        """
        Initialize a processor for handling streaming dictionary data.

        Args:
            save_path (str): Directory to save files to
            file_prefix (str): Prefix for saved files
            save_frequency (int): How often to save to disk (number of rows)
        """
        self.df = pd.DataFrame()
        self.save_path = save_path
        self.file_prefix = file_prefix
        self.save_frequency = save_frequency
        self.row_count = 0
        self.save_count = 0

    def process_row(
        self,
        row_dict: dict,
        input_text: str = None,
        ground_truth: str = None,
        benchmark_name: str = None,
        doc_id: str = None,
        write_to_disk: bool = True,
        write_to_history: bool = True
    ):
        """
        Process a single row of dictionary data.

        Args:
            row_dict (dict): Dictionary with structure like
                {'small': ([-2.58...], ['yes'], [False]), 'medium': ([-4.82...], ['yes'], [False])}
            input_text (str, optional): The input text for this row
            ground_truth (str, optional): The ground truth for this row
            benchmark_name (str, optional): The benchmark name for this row
            doc_id (str, optional): The document ID for this row
            write_to_disk (bool, optional): Whether to write samples to disk
            write_to_history (bool, optional): Whether to concatenate new rows to previous rows and return the entire df
        """
        # Create a flattened dictionary for this row
        flat_dict = {}

        # Add the two extra columns
        flat_dict["input_text"] = input_text
        flat_dict["ground_truth"] = ground_truth
        flat_dict["doc_id"] = doc_id

        # For each key in the dictionary
        for model_category, values in row_dict.items():
            for k, v in values.items():
                flat_dict[f"{model_category}_{k}"] = v[0] if type(v) is list else v

        # Convert the flattened dictionary to a DataFrame row and append
        row_df = pd.DataFrame([flat_dict])

        # Append to the main DataFrame
        if write_to_history:
            self.df = pd.concat([self.df, row_df], ignore_index=True)

            # Increment row counter
            self.row_count += 1

            # Save if we've reached the save frequency
            if self.row_count % self.save_frequency == 0 and write_to_disk:
                self.save_to_disk(benchmark_name=benchmark_name)

            return self.row_count

        else:
            return row_df

    def save_to_disk(self, final=False, benchmark_name: str = None):
        """
        Save the current DataFrame to disk.

        Args:
            final (bool): Whether this is the final save (affects filename)
            benchmark_name (str, optional): The benchmark name
        """
        if self.df.empty:
            return

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if final:
            filename = f"{self.file_prefix}final_{timestamp}.csv"
        else:
            self.save_count += 1
            filename = f"{self.file_prefix}batch_{self.save_count}_{timestamp}.csv"

        save_path = self.get_or_create_save_path(base_save_path=self.save_path, benchmark_name=benchmark_name)
        full_path = os.path.join(save_path, filename)

        # Save to CSV
        self.df.to_csv(full_path, index=False)
        print(f"Saved {len(self.df)} rows to {full_path}")

        return full_path

    def finalize(self, write_to_disk: bool = True):
        """
        Save any remaining data and return summary.
        write_to_disk (bool, optional): Whether to write samples to disk
        """

        # Save any remaining data that hasn't hit the save threshold
        if self.row_count % self.save_frequency != 0 and write_to_disk:
            final_path = self.save_to_disk(final=True)
        else:
            final_path = None

        return {
            "total_rows_processed": self.row_count,
            "save_batches": self.save_count,
            "final_save_path": final_path,
            "columns": list(self.df.columns)
        }

    @staticmethod
    def get_or_create_save_path(base_save_path: str, benchmark_name: str = None):

        if benchmark_name is None:
            benchmark_name = ""

        # Create save directory if it doesn't exist
        save_path = f"{base_save_path}/{benchmark_name}"
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        return save_path


class DataExtractor:

    def get_data(self, benchmark_name, request) -> Dict[str, str]:

        if benchmark_name == 'boolq':
            relevant_data = self.get_data_for_boolq(request)

        else:
            raise NotImplementedError(f"Data extractor for benchmark {benchmark_name} not implemented.")

        # Check for relevant keys
        assert "doc_id" in relevant_data.keys(), "Make sure 'doc_id' is part of the request."
        assert "input_data" in relevant_data.keys(), "Make sure 'input_data' is being extracted."
        assert "ground_truth" in relevant_data.keys(), "Make sure 'ground_truth' is being extracted."

        return relevant_data

    @staticmethod
    def get_data_for_boolq(request):
        """
            Extract the question and response from an Instance object.

            Args:
                request: The Instance object containing the document data

            Returns:
                dict: (question, response)
            """
        relevant_data = {
            "doc_id": request.doc_id,
            "input_data": request.doc['question'],
            "ground_truth": request.arguments[1].strip()
        }

        return relevant_data


class SampleGenerator:
    def make_boolq_sample(self, request: Instance, label_dict: dict):
        """
        Creates a pandas DataFrame for BoolQ format based on the provided input text and label dictionary.
        BoolQ asks the same question twice with different paragraphs and background information. One variant is true,
        the other one is false. When creating the "input_text" for our classifier, we need to concatenate the question
        and the ground_truth response. Otherwise, we will confuse the model.

        Args:
            request (Instance): The input request object provided by LM-Eval
            label_dict (dict): Dictionary with keys as label names and values as tuples of (log prob, label, correct)

        Returns:
            pd.DataFrame: DataFrame with columns for id, input_text, and a label column for each key in label_dict
        """

        # Create a dictionary to hold our data
        data = {
            "doc_id": request.doc_id,
            # This is the question plus the expected model output ("yes"/"no").
            "input_text": f"{request.doc['question']} -{request.arguments[1]}"
        }

        def get_value(val):
            if type(val) is list:
                out = get_value(val[0])
            elif type(val) is bool:
                out = int(val)
            else:
                out = val

            return out

        # Add a column for each label in the label_dict
        for model_category, model_outputs in label_dict.items():
            for name, value in model_outputs.items():
                data[f"{name}_{model_category}"] = get_value(value)

        # Create and return the DataFrame
        return pd.DataFrame([data])

    def make_sample(self, request: Instance, label_dict, benchmark_name):
        """
        Routes the input to the appropriate sample creation function based on benchmark_name.

        Args:
            input_text (str): The input text to include in the sample
            label_dict (dict): Dictionary with keys as label names and values as tuples
            benchmark_name (str): Name of the benchmark to determine routing

        Returns:
            pd.DataFrame: Properly formatted DataFrame based on the benchmark
        """
        if benchmark_name.lower() == "boolq":
            return self.make_boolq_sample(request, label_dict)
        else:
            # Default format or handle other benchmark types
            # For now, using the same format as BoolQ
            raise NotImplementedError(f"Sample creator for benchmark {benchmark_name} not implemented.")
