import wandb
import os
import json
import time
import pandas as pd
from tqdm.notebook import tqdm
import pickle
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler("wandb_download.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_file_path: str):
    """Load checkpoint data if it exists"""
    if os.path.exists(checkpoint_file_path):
        try:
            with open(checkpoint_file_path, 'rb') as f:
                checkpoint = pickle.load(f)
            logger.info(f"Loaded checkpoint: {len(checkpoint['completed_runs'])} runs already processed")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return {'completed_runs': set(), 'run_summaries': []}
    return {'completed_runs': set(), 'run_summaries': []}


def save_checkpoint(checkpoint, checkpoint_file_path: str):
    """Save checkpoint data"""
    folder_pth = "/".join(checkpoint_file_path.split("/")[:-1])
    Path(folder_pth).mkdir(parents=True, exist_ok=True)

    with open(checkpoint_file_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    logger.info(f"Checkpoint saved: {len(checkpoint['completed_runs'])} runs processed")


def download_run_history(run, output_dir):
    """Download full history for a single run"""
    run_dir = os.path.join(output_dir, run.id)
    os.makedirs(run_dir, exist_ok=True)

    # Get run summary
    summary = {k: v for k, v in run.summary._json_dict.items() if not k.startswith('_')}

    # Get config
    config = {k: v for k, v in run.config.items() if not k.startswith('_')}

    # Get metrics history
    history = run.scan_history()
    history_list = list(history)

    # Save data
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)

    # Save history as JSON and CSV
    if history_list:
        # Convert to pandas DataFrame for easier manipulation
        history_df = pd.DataFrame(history_list)

        # Save as CSV
        history_df.to_csv(os.path.join(run_dir, 'history.csv'), index=False)

        # Save as JSON
        history_df.to_json(os.path.join(run_dir, 'history.json'), orient='records', indent=2)

    # Save run metadata
    metadata = {
        'id': run.id,
        'name': run.name,
        'path': run.path,
        'project': run.project,
        'entity': run.entity,
        'url': run.url,
        'created_at': str(run.created_at),
        'tags': run.tags,
    }

    with open(os.path.join(run_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    return {
        'id': run.id,
        'name': run.name,
        'path': run.path,
        'url': run.url,
        'created_at': str(run.created_at),
        'tags': run.tags,
        'summary': summary,
        'config': config
    }


def download_log_data(entity: str, project_name: str, save_dir: str, batch_size: int = 50):
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_file_path=f"{save_dir}/checkpoints")
    completed_runs = checkpoint['completed_runs']
    run_summaries = checkpoint['run_summaries']

    # Initialize W&B API
    api = wandb.Api()

    # Get all runs from the project
    runs = api.runs(f"{entity}/{project_name}")
    logger.info(f"Found {len(runs)} total runs in project")

    # Create a progress bar for all runs
    with tqdm(total=len(runs), desc="Downloading runs") as pbar:
        # Update progress bar with already completed runs
        pbar.update(len(completed_runs))

        # Process runs in batches to avoid memory issues
        remaining_runs = [run for run in runs if run.id not in completed_runs]

        for i in range(0, len(remaining_runs), batch_size):
            batch = remaining_runs[i:i + batch_size]
            logger.info(f"Processing batch of {len(batch)} runs (starting at index {i})")

            for run in batch:
                try:
                    # Skip if already processed
                    if run.id in completed_runs or os.path.exists(f"{save_dir}/{run.id}"):
                        continue

                    # Download run data
                    logger.info(f"Downloading run {run.id}: {run.name}")
                    run_summary = download_run_history(run, save_dir)

                    # Update progress
                    run_summaries.append(run_summary)
                    completed_runs.add(run.id)
                    pbar.update(1)

                    # Save checkpoint after each run
                    checkpoint = {
                        'completed_runs': completed_runs,
                        'run_summaries': run_summaries
                    }
                    save_checkpoint(checkpoint, checkpoint_file_path=f"{save_dir}/checkpoints")

                except Exception as e:
                    logger.error(f"Error processing run {run.id}: {e}")
                    continue

    # Create a summary DataFrame
    summary_df = pd.DataFrame(run_summaries)
    summary_df.to_csv(os.path.join(save_dir, 'all_runs_summary.csv'), index=False)
    summary_df.to_json(os.path.join(save_dir, 'all_runs_summary.json'), orient='records', indent=2)

    logger.info(f"Download complete! Downloaded {len(completed_runs)} runs")
    logger.info(f"All data saved to {os.path.abspath(save_dir)}")

    # Optional: Delete checkpoint file after successful completion
    if os.path.exists(f"{save_dir}/checkpoints") and len(completed_runs) == len(runs):
        os.remove(f"{save_dir}/checkpoints")
        logger.info("Checkpoint file removed after successful completion")

    return summary_df


def load_all_histories_to_dataframe(data_dir: str):
    """
    Load all run histories from the data directory into a single DataFrame.

    Parameters:
    - data_dir: Directory where run data was saved (default: 'wandb_downloads')

    Returns:
    - combined_df: DataFrame containing all run histories with run metadata
    """
    # Get all run directories (exclude non-directory files like summaries)
    run_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    logger.info(f"Found {len(run_dirs)} run directories")

    all_runs_data = []

    for run_id in tqdm(run_dirs, desc="Loading run histories"):
        run_dir = os.path.join(data_dir, run_id)

        # Load metadata if available
        metadata = {}
        metadata_path = os.path.join(run_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        # Try to load history from CSV first (faster and more reliable)
        history_csv_path = os.path.join(run_dir, 'history.csv')
        history_json_path = os.path.join(run_dir, 'history.json')

        # Initialize history_df to None
        history_df = None

        # Try CSV first
        if os.path.exists(history_csv_path):
            try:
                history_df = pd.read_csv(history_csv_path)
            except Exception as e:
                logger.error(f"Error reading CSV history for run {run_id}: {e}")
                history_df = None

        # If CSV failed or doesn't exist, try JSON
        if history_df is None and os.path.exists(history_json_path):
            try:
                with open(history_json_path, 'r') as f:
                    history_data = json.load(f)
                history_df = pd.DataFrame(history_data)
            except Exception as e:
                logger.error(f"Error reading JSON history for run {run_id}: {e}")
                continue

        # If both failed or don't exist, skip this run
        if history_df is None or history_df.empty:
            logger.info(f"No valid history found for run {run_id}, skipping")
            continue

        # Add run metadata to each row
        history_df['run_id'] = run_id

        # Add other metadata if available
        if metadata:
            history_df['run_name'] = metadata.get('name', '')

            if 'tags' in metadata and metadata['tags']:
                history_df['run_tags'] = ', '.join(metadata['tags'])

            if 'created_at' in metadata:
                history_df['created_at'] = metadata['created_at']

        # Load config if available
        config_path = os.path.join(run_dir, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)

                # Add selected config parameters as columns
                for key, value in config.items():
                    # Skip internal keys
                    if key.startswith('_'):
                        continue

                    # Handle nested config values (common in W&B)
                    if isinstance(value, dict) and 'value' in value:
                        history_df[f'config_{key}'] = value['value']
                    else:
                        history_df[f'config_{key}'] = value
            except Exception as e:
                logger.info(f"Error loading config for run {run_id}: {e}")

        # Append to list of all runs
        all_runs_data.append(history_df)

    if not all_runs_data:
        logger.warn("No valid run data found!")
        return None

    # Combine all runs into a single DataFrame
    combined_df = pd.concat(all_runs_data, ignore_index=True)
    logger.info(f"Combined DataFrame shape: {combined_df.shape}")

    logger.info(f"Total rows: {len(combined_df)}")
    logger.info(f"Unique runs: {combined_df['run_id'].nunique()}")

    metrics = [col for col in combined_df.columns  if not col.startswith('run_') and not col.startswith('config_')]
    logger.info(f"Metrics available: {metrics}")

    return combined_df
