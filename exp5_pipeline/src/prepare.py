import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

input_file = "exp5_pipeline/data/raw/iris.csv"
output_dir = Path("exp5_pipeline/data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file)

train, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["species"]
)

train.to_csv(output_dir / "train.csv", index=False)
test.to_csv(output_dir / "test.csv", index=False)

print("Data preparation completed.")
print("Training rows:", len(train))
print("Testing rows:", len(test))
