import json
import pandas as pd
from src.config import CMDB_FILE


def load_cmdb() -> list[str]:
    """
    Reads app_cmdb.xlsx and converts each row into a
    descriptive plain-English sentence for embedding.
    """
    df    = pd.read_excel(CMDB_FILE)
    texts = []

    for _, row in df.iterrows():
        text = (
            f"{row['Application Name']} (ARN {row['ARN No']}) is a "
            f"{row['Category']} application in the {row['Business Vertical']} vertical. "
            f"Status: {row['Application Status']}. "
            f"Tier: {row['Application Tier']}. "
            f"Criticality: {row['Business Criticality']}. "
            f"Owner: {row['App Owner']}. "
            f"Hosted on: {row['Hosted on']}. "
            f"Infrastructure: {row['Infrastructure']}. "
            f"RTO: {row['Recovery Time Objective(RTO)']}. "
            f"RPO: {row['Recovery Point Objective(RPO)']}. "
            f"Description: {row['Description']}."
        )
        texts.append(text)

    return texts


def load_graph_json(filepath: str) -> list[str]:
    """
    Reads a graphrag JSON file and converts each record
    into a readable key-value string for embedding.
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    return [
        ", ".join(f"{k}: {v}" for k, v in item.items() if v)
        for item in data
    ]