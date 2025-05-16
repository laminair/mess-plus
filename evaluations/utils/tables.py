def build_pivot_table_for_main_results(input_df: pd.DataFrame, model_cols: list):

	latest_steps = input_df.groupby(['benchmark_name', 'alpha'])['_step'].transform('max')
	is_last_step = input_df['_step'] == latest_steps

	# Create new columns with the final values
	for col in model_cols:
	    final_values = input_df.loc[is_last_step, ['benchmark_name', 'alpha', col]]
	    final_values = final_values.drop_duplicates(['benchmark_name', 'alpha'])
	    input_df = pd.merge(
	        input_df,
		    final_values.rename(columns={col: f"final_{col}"}),
	        on=['benchmark_name', 'alpha'],
	        how='left'
	    )

	# Add the final model values to the pivot table
	merged_pvt_table = input_df.loc[:, ["benchmark_name", "alpha", "V", "avg_accuracy", "mess_plus/energy"] + [f"final_{col}" for col in model_cols]].pivot_table(
	    index=["benchmark_name", "alpha"],
	    columns=["V"],
	    values=["avg_accuracy", "mess_plus/energy"] + [f"final_{col}" for col in model_cols],
	    aggfunc={
	        "avg_accuracy": ["mean", "std"],
	        "mess_plus/energy": ["sum", "std"],
	        **{f"final_{col}": ['first'] for col in model_cols}
	    }
	)

	return merged_pvt_table

