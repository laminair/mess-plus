import logging
import pandas as pd
import os

from datetime import datetime
from lm_eval.api.task import Instance, Task

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
        self.benchmark_name = None

    def process_row(
        self,
        sample: pd.DataFrame,
        benchmark_name: str,
        write_to_disk: bool = True
    ):
        self.benchmark_name = benchmark_name
        self.df = pd.concat([self.df, sample], ignore_index=True)

        # Increment row counter
        self.row_count += 1

        # Save if we've reached the save frequency
        if self.row_count % self.save_frequency == 0 and write_to_disk:
            self.save_to_disk(benchmark_name=benchmark_name)

        return self.row_count

    def save_to_disk(self, final=False, benchmark_name: str = None):
        """
        Save the current DataFrame to disk.

        Args:
            final (bool): Whether this is the final save (affects filename)
            benchmark_name (str, optional): The benchmark name
        """
        if self.df.empty:
            return

        # Set benchmark name to place final chunk of data into the right folder
        self.benchmark_name = benchmark_name

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
        logging.info(f"Saved {len(self.df)} rows to {full_path}")

        return full_path

    def finalize(self):
        """
        Save any remaining data and return summary.
        write_to_disk (bool, optional): Whether to write samples to disk
        """

        # Save any remaining data that hasn't hit the save threshold
        if self.row_count % self.save_frequency != 0:
            final_path = self.save_to_disk(final=True, benchmark_name=self.benchmark_name)
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

    def __init__(self):
        self.benchmark_metrics_mapping = {
            "boolq": "acc"
        }

    def make_boolq_sample(self, doc_id, input_data, model_response_data, stage):

        # Create a dictionary to hold our data
        data = {
            "doc_id": doc_id,
            # This is the question plus the expected model output ("yes"/"no").
            "input_text": f"{input_data['question']}"
        }

        if stage == "train" and model_response_data is not None:
            metric_name = self.benchmark_metrics_mapping["boolq"]
            for model_category in model_response_data.keys():
                data.update({
                    f"benchmark_name": "boolq",
                    f"label_{model_category}": model_response_data[model_category][metric_name],
                    f"{metric_name}_{model_category}": model_response_data[model_category][metric_name],
                    f"energy_consumption_{model_category}": model_response_data[model_category]["energy_consumption"],
                    f"inference_time_{model_category}": model_response_data[model_category]["inference_time"],
                })

        # Create and return the DataFrame
        return pd.DataFrame([data])

    def make_sample(
        self,
        doc_id: int,
        input_data: dict,
        task: Task,
        model_response_data: dict = None,
        stage: str = "train"
    ):

        if task.config.task.lower() == "boolq":
            return self.make_boolq_sample(doc_id, input_data, model_response_data, stage)
        else:
            # Default format or handle other benchmark types
            # For now, using the same format as BoolQ
            raise NotImplementedError(f"Sample creator for benchmark {task.config.task.lower()} not implemented.")
