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
        self.last_saved_row = 0  # Track the last row we saved

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
        Save the current batch of DataFrame to disk.
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

        # Calculate which rows to save (only the new ones since last save)
        current_batch = self.df.iloc[self.last_saved_row:]

        # Save only the current batch to CSV
        current_batch.to_csv(full_path, index=False)

        # Update the last saved row count
        self.last_saved_row = len(self.df)

        logging.info(f"Saved {len(current_batch)} rows to {full_path}")
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


class SampleGenerator:

    def __init__(self):
        self.benchmark_metrics_mapping = {
            "boolq": "acc",
            "piqa": "acc",
            "logiqa": "acc",
            "logiqa2": "acc",
            "social_iqa": "acc",
            "triviaqa": "exact_match",
            "sciq": "acc",
            "arc_easy": "acc_norm",
            "arc_challenge": "acc_norm",
            "winogrande": "acc",
            "mmlu": "acc",
            "lambada_standard": "acc"
        }

    def make_sample_inner(self, doc_id, input_text, model_response_data, stage, benchmark_metric, benchmark_name: str = "boolq"):

        # Create a dictionary to hold our data
        data = {
            "doc_id": doc_id,
            # This is the question plus the expected model output ("yes"/"no").
            "input_text": input_text
        }

        if stage == "train" and model_response_data is not None:
            for model_category in model_response_data.keys():
                data.update({
                    f"benchmark_name": benchmark_name,
                    f"label_{model_category}": model_response_data[model_category][benchmark_metric],
                    f"{benchmark_metric}_{model_category}": model_response_data[model_category][benchmark_metric],
                    f"energy_consumption_{model_category}": model_response_data[model_category]["energy_consumption"],
                    f"inference_time_{model_category}": model_response_data[model_category]["inference_time"],
                })

        # Create and return the DataFrame
        return pd.DataFrame([data])

    def make_boolq_sample(self, doc_id, input_data, model_response_data, stage, benchmark_metric):

        return self.make_sample_inner(
            doc_id,
            input_text=f"{input_data['question']}",
            model_response_data=model_response_data,
            stage=stage,
            benchmark_name="boolq",
            benchmark_metric=benchmark_metric,
        )

    def make_piqa_sample(self, doc_id, input_data, model_response_data, stage, benchmark_metric):
        return self.make_sample_inner(
            doc_id,
            input_text=f"{input_data['goal']}",
            model_response_data=model_response_data,
            stage=stage,
            benchmark_name="piqa",
            benchmark_metric=benchmark_metric
        )

    def make_logiqa_sample(self, doc_id, input_data, model_response_data, stage, benchmark_metric):
        return self.make_sample_inner(
            doc_id,
            input_text=f"{input_data['question']}",
            model_response_data=model_response_data,
            stage=stage,
            benchmark_name="logiqa",
            benchmark_metric=benchmark_metric
        )

    def make_logiqa2_sample(self, doc_id, input_data, model_response_data, stage, benchmark_metric):
        return self.make_sample_inner(
            doc_id,
            input_text=f"{input_data['question']}",
            model_response_data=model_response_data,
            stage=stage,
            benchmark_name="logiqa2",
            benchmark_metric=benchmark_metric
        )

    def make_socialiqa_sample(self, doc_id, input_data, model_response_data, stage, benchmark_metric):
        return self.make_sample_inner(
            doc_id,
            input_text=f"{input_data['context']}{input_data['question']}",
            model_response_data=model_response_data,
            stage=stage,
            benchmark_name="social_iqa",
            benchmark_metric=benchmark_metric
        )

    def make_triviaqa_sample(self, doc_id, input_data, model_response_data, stage, benchmark_metric):
        return self.make_sample_inner(
            doc_id,
            input_text=f"{input_data['question']}",
            model_response_data=model_response_data,
            stage=stage,
            benchmark_name="triviaqa",
            benchmark_metric=benchmark_metric
        )

    def make_sciq_sample(self, doc_id, input_data, model_response_data, stage, benchmark_metric):
        return self.make_sample_inner(
            doc_id,
            input_text=f"{input_data['question']}",
            model_response_data=model_response_data,
            stage=stage,
            benchmark_name="sciq",
            benchmark_metric=benchmark_metric
        )

    def make_arc_easy_sample(self, doc_id, input_data, model_response_data, stage, benchmark_metric):
        return self.make_sample_inner(
            doc_id,
            input_text=f"{input_data['question']}",
            model_response_data=model_response_data,
            stage=stage,
            benchmark_name="arc_easy",
            benchmark_metric=benchmark_metric
        )

    def make_arc_challenge_sample(self, doc_id, input_data, model_response_data, stage, benchmark_metric):
        return self.make_sample_inner(
            doc_id,
            input_text=f"{input_data['question']}",
            model_response_data=model_response_data,
            stage=stage,
            benchmark_name="arc_challenge",
            benchmark_metric=benchmark_metric
        )

    def make_winogrande_sample(self, doc_id, input_data, model_response_data, stage, benchmark_metric):
        return self.make_sample_inner(
            doc_id,
            input_text=f"{input_data['sentence']}",
            model_response_data=model_response_data,
            stage=stage,
            benchmark_name="winogrande",
            benchmark_metric=benchmark_metric
        )

    def make_mmlu_sample(self, doc_id, input_data, model_response_data, stage, benchmark_metric, benchmark_name):
        # TODO: We may want to include the subject at some point for improved embeddings.
        return self.make_sample_inner(
            doc_id,
            input_text=f"{input_data['question']}",
            model_response_data=model_response_data,
            stage=stage,
            benchmark_name=benchmark_name,
            benchmark_metric=benchmark_metric
        )

    def make_lambada_sample(self, doc_id, input_data, model_response_data, stage, benchmark_metric, benchmark_name):
        return self.make_sample_inner(
            doc_id,
            input_text=f"{input_data['text']}",
            model_response_data=model_response_data,
            stage=stage,
            benchmark_name=benchmark_name,
            benchmark_metric=benchmark_metric
        )

    def make_sample(
        self,
        doc_id: int,
        input_data: dict,
        benchmark_metric: str,
        task: Task,
        model_response_data: dict = None,
        stage: str = "train"
    ):

        if task.config.task.lower() == "boolq":
            return self.make_boolq_sample(doc_id, input_data, model_response_data, stage, benchmark_metric)
        elif task.config.task.lower() == "piqa":
            return self.make_piqa_sample(doc_id, input_data, model_response_data, stage, benchmark_metric)
        elif task.config.task.lower() == "logiqa":
            return self.make_logiqa_sample(doc_id, input_data, model_response_data, stage, benchmark_metric)
        elif task.config.task.lower() == "logiqa2":
            return self.make_logiqa_sample(doc_id, input_data, model_response_data, stage, benchmark_metric)
        elif task.config.task.lower() == "social_iqa":
            return self.make_socialiqa_sample(doc_id, input_data, model_response_data, stage, benchmark_metric)
        elif task.config.task.lower() == "triviaqa":
            return self.make_triviaqa_sample(doc_id, input_data, model_response_data, stage, benchmark_metric)
        elif task.config.task.lower() == "sciq":
            return self.make_sciq_sample(doc_id, input_data, model_response_data, stage, benchmark_metric)
        elif task.config.task.lower() == "arc_easy":
            return self.make_arc_easy_sample(doc_id, input_data, model_response_data, stage, benchmark_metric)
        elif task.config.task.lower() == "arc_challenge":
            return self.make_arc_challenge_sample(doc_id, input_data, model_response_data, stage, benchmark_metric)
        elif task.config.task.lower() == "winogrande":
            return self.make_winogrande_sample(doc_id, input_data, model_response_data, stage, benchmark_metric)
        elif "mmlu" in task.config.task.lower():
            return self.make_mmlu_sample(doc_id, input_data, model_response_data, stage, benchmark_metric, benchmark_name=task.config.task.lower())
        elif "lambada" in task.config.task.lower():
            return self.make_lambada_sample(doc_id, input_data, model_response_data, stage, benchmark_metric, benchmark_name=task.config.task.lower())
        else:
            # Default to get all relevant information when setting up a new benchmark.
            print("INPUT_DATA: ", input_data)
            print("MODEL_RESPONSE_DATA: ", model_response_data)
            raise NotImplementedError(
                f"Sample creator for benchmark {task.config.task.lower()} not implemented. "
                f"Check sample data above to see how you need to configure the sample generator for your benchmark.")
