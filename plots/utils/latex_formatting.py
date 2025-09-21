import pandas as pd
import numpy as np


def dataframe_to_latex_with_color(df, high_color='green!60', low_color='red!60'):
	"""
	Convert a pandas DataFrame to a LaTeX table with color coding for accuracy columns.

	Parameters:
	df: pandas DataFrame (must have 'alpha' in the index or as a column)
	high_color: LaTeX color for values >= threshold
	low_color: LaTeX color for values < threshold
	"""

	# Create a copy to avoid modifying the original
	df_copy = df.copy()

	# Convert energy values to MJ (divide by 1,000,000)
	if isinstance(df_copy.columns, pd.MultiIndex):
		for col in df_copy.columns:
			col_name = " ".join(str(c) for c in col).lower()
			if 'energy' in col_name or 'consumption' in col_name or 'mess_plus' in col_name:
				df_copy[col] = df_copy[col] / 1000000
	else:
		for col in df_copy.columns:
			if 'energy' in col.lower() or 'consumption' in col.lower() or 'mess_plus' in col.lower():
				df_copy[col] = df_copy[col] / 1000000

	# Round all numeric values to 2 decimal places
	df_rounded = df_copy.round(2)

	# Function to determine if a value should be colored
	def should_color(col_name):
		# Color if 'accuracy' is in the column name (case insensitive)
		return 'accuracy' in col_name.lower() or 'label' in col_name.lower()

	# Function to format a cell with color coding
	def format_cell(value, col_name, alpha):
		if should_color(col_name) and pd.notna(value):
			if value >= alpha:
				return f"\\cellcolor{{{high_color}}}{value:.2f}"
			else:
				return f"\\cellcolor{{{low_color}}}{value:.2f}"
		else:
			if pd.notna(value):
				return f"{value:.2f}"
			else:
				return "-"

	# Start building the LaTeX table
	latex_lines = []
	latex_lines.append("\\begin{table}[h!]")
	latex_lines.append("\\centering")
	latex_lines.append("\\begin{tabular}{l|" + "c" * (len(df.columns)) + "}")
	latex_lines.append("\\toprule")

	# Handle multi-level columns
	if isinstance(df.columns, pd.MultiIndex):
		# Get the number of levels
		n_levels = df.columns.nlevels

		# Create header rows for each level
		for level in range(n_levels):
			header_row = []
			if level == 0:
				header_row.append("\\multirow{2}{*}{Benchmark}")
			else:
				header_row.append("")

			# Get labels for this level
			labels = []
			spans = []
			current_label = None
			current_span = 0

			for col in df.columns:
				label = str(col[level])
				if label != current_label:
					if current_label is not None:
						labels.append((current_label, current_span))
					current_label = label
					current_span = 1
				else:
					current_span += 1

			# Don't forget the last group
			if current_label is not None:
				labels.append((current_label, current_span))

			# Add multicolumn headers
			for label, span in labels:
				# Add (MJ) to energy-related headers
				if any(energy_term in str(label).lower() for energy_term in ['energy', 'consumption', 'mess_plus']):
					label = f"{label} (MJ)"

				if span > 1:
					header_row.append(f"\\multicolumn{{{span}}}{{c}}{{{label}}}")
				else:
					header_row.append(label)

			latex_lines.append(" & ".join(header_row) + " \\\\")

		# Add horizontal line between header levels
		latex_lines.append("\\cmidrule{2-" + str(len(df.columns) + 1) + "}")
	else:
		# Simple single-level header
		header_row = ["Benchmark"]
		for col in df.columns:
			col_str = str(col)
			# Add (MJ) to energy-related headers
			if any(energy_term in col_str.lower() for energy_term in ['energy', 'consumption', 'mess_plus']):
				col_str = f"{col_str} (MJ)"
			header_row.append(col_str)
		latex_lines.append(" & ".join(header_row) + " \\\\")

	latex_lines.append("\\midrule")

	# Add data rows
	for idx, row in df_rounded.iterrows():
		# Handle MultiIndex row
		if isinstance(idx, tuple):
			benchmark_name = idx[0]
			alpha = idx[1] if len(idx) > 1 else None
		else:
			benchmark_name = str(idx)
			alpha = None

		# If alpha is not in the index, check if it's a column
		if alpha is None:
			if 'alpha' in df.columns:
				alpha = row['alpha']
			else:
				# If alpha is not found, raise an error
				raise ValueError("Alpha must be either in the index or as a column")

		# Format the row
		row_data = [benchmark_name]
		for col_name, value in row.items():
			# Get the actual column name for color checking
			if isinstance(df.columns, pd.MultiIndex):
				actual_col_name = " ".join(str(c) for c in col_name)
			else:
				actual_col_name = str(col_name)

			formatted_value = format_cell(value, actual_col_name, alpha)
			row_data.append(formatted_value)

		latex_lines.append(" & ".join(row_data) + " \\\\")

	latex_lines.append("\\bottomrule")
	latex_lines.append("\\end{tabular}")

	# Add caption and label
	latex_lines.append("\\caption{Comparison of Model Performance}")
	latex_lines.append("\\label{tab:model_comparison}")
	latex_lines.append("\\end{table}")

	# Add required packages in comments
	packages = [
		"% Required LaTeX packages:",
		"% \\usepackage{booktabs}",
		"% \\usepackage{multirow}",
		"% \\usepackage{xcolor}",
		"% \\usepackage{colortbl}",
		""
	]

	return "\n".join(packages + latex_lines)
