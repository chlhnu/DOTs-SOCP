"""Log file parser and table exporter.

This script parses log files generated during project execution to extract running metrics
like execution time and convergence information. The parsed data is organized into a table
and can be exported in various formats (.tex, .html, .csv, .xlsx).

Usage:
    python log2table.py -i <log_file> -o <output_file(s)>
Args:
    --input (-i): Path to the log file to parse
    --output (-o): One or more output file paths (.tex/.html/.csv/.xlsx)
"""

__all__ = ['log2table']

import re
import pandas as pd
import warnings
from argparse import ArgumentParser, RawTextHelpFormatter

import sys
from pathlib import Path
root_of_import = Path(__file__).parent.parent
if str(root_of_import) not in sys.path:
    sys.path.insert(0, str(root_of_import))

from dot_surface_socp.utils.file_process import export_table

def match_the_line(file_lines, pattern, end_pattern, idx):
    """Match the pattern until the end_pattern from the beginning idx
    """
    value = None

    while idx < len(file_lines) and not end_pattern.match(file_lines[idx]):
        match_res = pattern.search(file_lines[idx])
        if match_res:
            value = match_res.group(1)
            break
        idx += 1

    return value, idx

def parse_log(file_path, reg_start, reg_content_arr, sequential_matching: bool = False):
    """Parse the log file

    Usage
    1. One-by-one Matching
        >>> parse_log(file_path, reg_start, reg_content_arr)
    2. Sequential matching
        Fast, but require that the order of <reg_content_seq> is just the order in which the content appears in the log file
        >>> parse_log(file_path, reg_start, reg_content_arr, sequential_matching=True)
    """
    start_pattern = re.compile(reg_start)
    content_pattern_arr = [re.compile(reg_content_arr[idx]) for idx in range(len(reg_content_arr))]

    with open(file_path, 'r') as f:
        lines = f.readlines()

    line_block = 0
    contents = []
    while line_block < len(lines):
        next_block = line_block + 1

        # Matching the block
        if start_pattern.match(lines[line_block]):
            line_content = line_block + 1

            # Matching the content
            item = [None] * len(content_pattern_arr)
            for idx_content in range(len(content_pattern_arr)):
                val, line_matched = match_the_line(lines, content_pattern_arr[idx_content], start_pattern, line_content)
                item[idx_content] = val
                next_block = max(next_block, line_matched)

                if sequential_matching:
                    line_content = line_matched + 1 # continue to search

            if all(val is not None for val in item):
                contents.append(item)
            else:
                warnings.warn(
                    f"An unexpected matched result (started with line {line_block}):\n"
                    f"Matched result: {item}\n"
                )

        line_block = next_block

    return contents

def log2table(file_path, out_tables):
    """Read the log file and output a table (or tables).
    
    Args:
        file_path (str): Path to the log file to parse.
        out_tables (str or list of str): Output file path(s) for the table(s). Supported formats: .tex, .html, .csv, .xlsx.
    """
    # ==== Custom the table based on the log file
    tag_start = r".*Info: Experiment Setting.*"
    tag_content_seq = [
        r"^Example name:\s*(\S+)",
        r"^Number of vertices:\s*(\d+)",
        r"^Number of triangles:\s*(\d+)",
        r"^Transportation cost:\s*([-+]?\d+\.\d+e[-+]?\d+)",
        r"^Time of steps\s*:\s*(\d+\.?\d*)\s*sec",
        r"^Total Iteration(?:\s*\(l\.l\.\))?\s*:\s*(\d+) iterations"
    ]
    content_names = ["Example", "Vertices", "Triangles", "Transport Cost", "Time [seconds]", "Iterations"]
    row_header = content_names[0]  # "Example"
    shared_col_header = [content_names[idx] for idx in [1, 2]]  # ["Vertices", "Triangles"]
    isolated_col_header = [content_names[idx] for idx in [5, 4, 3]]  # ["Iterations", "Time [seconds]", "Transport Cost"]
    # ==== End of custom table definition

    def apply_numeric_formatting_to_isolated_col(df, col_name, float_digits=4):
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce').round(float_digits)
    
    numeric_formatting_transport_cost = lambda df: apply_numeric_formatting_to_isolated_col(df, content_names[3], float_digits=4)

    # Parse log
    data = parse_log(file_path, tag_start, tag_content_seq)
    
    if not data:
        print(f"Warning: No data found in {file_path}")
        return

    df = pd.DataFrame(data, columns=content_names)
    df_combined_share_data = df.groupby(row_header)[shared_col_header].first()
    df_isolated_data = df.groupby(row_header)[isolated_col_header].first()
    numeric_formatting_transport_cost(df_isolated_data)
    
    df_table = pd.concat([df_combined_share_data, df_isolated_data], axis=1)
    df_table.reset_index(inplace=True)
    df_table[row_header] = df_table[row_header].str.replace('_', ' ').str.title()

    # Export table
    if isinstance(out_tables, str):
        export_table(df_table, out_tables)
    elif isinstance(out_tables, list) and all(isinstance(item, str) for item in out_tables):
        for idx in range(len(out_tables)):
            export_table(df_table, out_tables[idx])

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Read logging file and output table(s).",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("-i", "--input", help="Path to logging file")
    parser.add_argument("-o", "--output", nargs="+", help="List of output files.\nSupported table formats: .tex, .html, .csv, .xlsx", required=True)
    args = parser.parse_args()

    log2table(file_path=args.input, out_tables=args.output)
