"""Inspect FakeMusicCaps metadata without downloading audio recordings."""

from collections import Counter

from datasets import load_dataset


def main():
    # This dataset contains metadata and audio file paths, not audio recordings.
    dataset = load_dataset("DeepFense/FakeMusicCaps")
    print("Dataset: DeepFense/FakeMusicCaps")
    print("Available splits:", ", ".join(dataset.keys()))

    for split_name, split in dataset.items():
        print(f"\nSplit: {split_name}")
        print(f"Number of examples: {len(split):,}")
        print("Columns:", ", ".join(split.column_names))
        print("Features:", split.features)

        if "label" in split.column_names:
            print("Label distribution:")
            for label, count in Counter(split["label"]).items():
                print(f"  {label}: {count:,}")
        else:
            print("No label column available.")

        print("Sample records (up to 3):")
        for record in split.select(range(min(3, len(split)))):
            print(record)


if __name__ == "__main__":
    main()
