import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pathlib import Path

BASE_DIR = Path("Output")
RUN_TAG = "run1"  # Update to match the run you are reconstructing

def rebuild_missing_finals():
    for place_dir in BASE_DIR.iterdir():
        if not place_dir.is_dir():
            continue
   
        safe_name = place_dir.name
        final_file = place_dir / f"{safe_name}_final_{RUN_TAG}.parquet"

        if final_file.exists():
            print(f"[SKIP] Final file exists for {safe_name}")
            continue

        tick_files = list(place_dir.glob(f"{safe_name}_tick*_rank*.parquet"))
        if not tick_files:
            print(f"[SKIP] No tick files found for {safe_name}")
            continue

        print(f"[BUILD] Merging {len(tick_files)} files for {safe_name}")

        # Load as dataset and sort
        dataset = ds.dataset(tick_files, format="parquet")
        table = dataset.to_table().sort_by([("p_id", "ascending"), ("tick", "ascending")])

        # Save final merged file
        pq.write_table(table, final_file, compression="snappy")

        # Delete old tick files
        for f in tick_files:
            f.unlink()

        print(f"[DONE] Final file written and intermediates deleted for {safe_name}")

if __name__ == "__main__":
    rebuild_missing_finals()
