from utils.load_data import build_spectrum_graph_dataset
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Building dataset"
    )

    parser.add_argument(
        "--edge_info",
        type=str,
        help="Edge information to use to build the dataset, one of 'type', 'stereo' or 'mix'",
    )
    args = parser.parse_args()


    build_spectrum_graph_dataset(
        edge_info=args.edge_info, file_name=f"spectrum_graph_dataset_bond{args.edge_info}.pickle"
    )
