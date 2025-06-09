from pathlib import Path
from pandas import DataFrame
from dot_surface_socp.utils.type import CheckpointsErrorData


def export_table_from_checkpoints_error(
        error_checkpoints: list[CheckpointsErrorData],
        out: str
    ):
    """Export a table based on the checkpoints of error data.
    
    Parameters
    ----------
    error_checkpoints : list[CheckpointsErrorData]
        List of error data at each checkpoint.
    out : str
        Output file path for the table.
    """
    # Extract data from checkpoints
    data = [
        {
            'iteration': cp['iteration'],
            'time': cp['time'],
            'kkt_error': cp['kkt_error'],
            'l1_error': cp['error']['l1'],
            'l2_error': cp['error']['l2'],
            'linf_error': cp['error']['linf']
        }
        for cp in error_checkpoints
    ]

    # Create table
    df = DataFrame(data)
    df = df.sort_values('iteration')

    # Format table
    df[['l1_error', 'l2_error', 'linf_error', 'kkt_error']] = \
        df[['l1_error', 'l2_error', 'linf_error', 'kkt_error']].map(lambda x: '{:.2e}'.format(x))
    df['time'] = df['time'].map(lambda x: '{:.2f}'.format(x))
    df = df[['l1_error', 'l2_error', 'linf_error', 'kkt_error', 'iteration', 'time']]
    df.columns = ['L1', 'L2', 'L-Inf', 'KKT', 'Iteration', 'Time (s)']
    
    # Export table
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    export_table(df, out_path=out)


def export_table(dataframe: DataFrame, out_path: str):
    out_path = Path(out_path)
    file_extension = out_path.suffix

    exported_opts = {"index": False, "index_names": False}

    # Either `df.to_csv` or `df.to_excel` has no arg `index_names`
    if file_extension in [".csv", ".xlsx"]:
        exported_opts.pop("index_names")

    match file_extension:
        case ".tex":
            dataframe.to_latex(out_path, **exported_opts)
            print(f"Exported to LaTeX format: {out_path}")
        case ".html":
            dataframe.to_html(out_path, **exported_opts)
            print(f"Exported to HTML format: {out_path}")
        case ".csv":
            dataframe.to_csv(out_path, **exported_opts)
            print(f"Exported the CSV format: {out_path}")
        case ".xlsx":
            if dataframe.columns.nlevels > 1:
                # Export table with row index to avoid the following error:
                #   NotImplementedError: Writing to Excel with MultiIndex columns and no index ('index'=False) is not yet implemented.
                exported_opts["index"] = True

            dataframe.to_excel(out_path, **exported_opts)
            print(f"Exported to Excel format: {out_path}")
        case _:
            print(f"Unsupported file format: {file_extension}")