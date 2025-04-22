import os

from pathlib import Path


def write_figure_to_disk(plt, file_name, chapter_name=None, file_type="pdf", dpi=300):
    evaluations_dir = Path(f"{os.path.dirname(os.path.realpath(__file__))}")
    paper_dir = f"{Path(evaluations_dir).parent}/paper/"

    if chapter_name is None:
        chapter_name = "misc"

    save_path = Path(f"{paper_dir}/figures/{chapter_name}")
    save_path.mkdir(exist_ok=True, parents=True)
    save_path = Path(f"{save_path}/{file_name}.{file_type}")

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print("Plot saved.")
