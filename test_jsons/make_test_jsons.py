import json
from pathlib import Path

import pandas as pd


def to_json(row):
    row = row.to_dict()
    row["visit_date"] = row["visit_date"].isoformat()
    row["visit_time"] = row["visit_time"].isoformat()
    row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    return row


def main():
    project_dir = Path(__file__).resolve().parent.parent
    data_path = project_dir / 'data' / 'processed' / 'sessions_target.pkl'
    df = pd.read_pickle(data_path)

    # Для получения более контрастных тестовых представителей классов
    # дополнительно отфильтруем по наиболее информативному признаку utm_source

    row0 = df[
        (df['target'] == 0) &
        (df['utm_source'] == 'PlbkrSYoHuZBWfYjYnfw')
        ].iloc[0]

    row1 = df[
        (df['target'] == 1) &
        (df['utm_source'] == 'aXQzDWsJuGXeBXexNHjc')
        ].iloc[0]

    with open(project_dir / 'test_jsons' / 'predict_class0.json', 'w', encoding="utf-8") as f:
        json.dump(to_json(row0), f, ensure_ascii=False, indent=2, allow_nan=False)

    with open(project_dir / 'test_jsons' / 'predict_class1.json', 'w', encoding="utf-8") as f:
        json.dump(to_json(row1), f, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
