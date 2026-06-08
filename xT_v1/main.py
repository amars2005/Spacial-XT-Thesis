# main.py
import argparse
import os
from builder import DatasetBuilder
from model import ExpectedThreatModel

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="xT_v1 Random Forest pipeline")
    parser.add_argument("--build",      action="store_true",
                        help="Build the chained-event dataset CSV (run once)")
    parser.add_argument("--train",      action="store_true",
                        help="Train Random Forest on the full dataset")
    parser.add_argument("--subset360",  action="store_true",
                        help="Train and evaluate RF on the 360-data subset only "
                             "(uses identical train/test split as V2/V3)")
    parser.add_argument("--save-preds", action="store_true",
                        help="Save per-event probabilities/labels/match_ids to "
                             "xT_v1/predictions/preds.npz (requires --subset360)")
    args = parser.parse_args()

    # Default: run full pipeline if no flag given
    if not any([args.build, args.train, args.subset360]):
        args.build = True
        args.train = True

    if args.build:
        builder = DatasetBuilder()
        print("Starting data extraction...")
        raw_data = builder.build_dataset()
        if raw_data.empty:
            print("No data found. Exiting.")
            return
        final_data = builder.process_chains(raw_data)
        output_filename = os.path.join(_ROOT, "statsbomb_chained_dataset.csv")
        final_data.to_csv(output_filename, index=False)
        print("\n--- DONE ---")
        print(f"Data saved to {output_filename}")
        print(f"Total Rows: {len(final_data)}")
        print(f"Columns: {list(final_data.columns)}")

    if args.train:
        xt_model = ExpectedThreatModel()
        df_with_preds = xt_model.train()
        xt_model.visualize_value_map(df_with_preds)
        df_with_preds.to_csv(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_model_output.csv"),
            index=False,
        )

    if args.subset360:
        print("\n=== V1-RF: 360-DATA SUBSET EVALUATION ===")
        xt_model = ExpectedThreatModel()
        xt_model.train_360_subset(save_preds=args.save_preds)


if __name__ == "__main__":
    main()
