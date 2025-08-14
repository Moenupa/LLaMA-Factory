import json
import os
import os.path
import pickle
import random
import pandas as pd
import argparse


def read_file(file_path: str, **kwargs) -> pd.DataFrame:
    """a cached reading function for `pd.read_excel`"""
    pkl_path = f"{file_path}.pkl"

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    data = pd.read_excel(file_path, sheet_name=0, **kwargs)
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    return data


def format_data(
    df: pd.DataFrame, rag: str = "", data_format: str = "alpaca"
) -> list[dict]:
    # make sure rag data has formatting already
    assert rag == "" or rag.endswith("\n\n")

    if data_format == "alpaca":
        return [
            {
                "instruction": f"{rag}根据以上参考资料及案件信息，判断被告人应当判处的有期徒刑月数。",
                "input": f"{data.本院查明}\n\n{data.本院认为}\n\n判处被告人：",
                "output": f"{data.有期徒刑}个月有期徒刑。",
            }
            for data in df.itertuples()
        ]
    elif data_format == "math12k":
        # output is digit only, like math12k
        return [
            {
                "instruction": f"{rag}根据以上参考资料及案件信息，判断被告人应当判处的有期徒刑月数。",
                "input": f"{data.本院查明}\n\n{data.本院认为}\n\n判处被告人有期徒刑月数：",
                # "output": f"{data.有期徒刑}",
                "output": f"{data.有期徒刑}",
            }
            for data in df.itertuples()
        ]
    elif data_format == "easyr1":
        # for tuning on EasyR1
        return [
            {
                "problem": f"{rag}{data.本院查明}\n\n{data.本院认为}\n\n判处被告人有期徒刑月数：",
                "answer": f"{data.有期徒刑}",
            }
            for data in df.itertuples()
        ]
    else:
        raise NotImplementedError(f"Unsupported format: {data_format}")


def to_json(dataset: dict[str, list[dict]]) -> None:
    assert "train" in dataset
    assert "validation" in dataset
    assert "test" in dataset

    for n, data in dataset.items():
        file_path = f"{n}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent="\t")

        print(file_path, len(data))


def preprocess_xlsx(
    feat_path: str = "Shandong_feature_414.xlsx", data_path: str = "Shandong.xlsx"
) -> pd.DataFrame:
    feature_file = read_file(feat_path)

    # 删掉单被告为0的行 删掉数罪并罚为1的行
    feature_file = feature_file[feature_file["数罪并罚"] != 1]
    feature_file = feature_file[feature_file["单被告"] != 0]
    data_file = read_file(data_path)

    # 剩下的行为最终用于模型训练的数据:
    # X为右侧“山东省”文件的本院查明和本院认为列包括的文本
    # Y为左侧“山东省 feature”文件中的有期徒刑列
    # 上面X,Y可以通过案号匹配起来，做成一个文件

    data_file = data_file[["案号", "本院查明", "本院认为", "裁判结果"]]
    feature_file = feature_file[["案号", "有期徒刑"]]
    final_file = feature_file.merge(data_file, on="案号", how="inner")
    # final_file.to_csv("final_data.csv", index=False)

    final_file = final_file[final_file["有期徒刑"] <= 36]
    final_file = final_file[final_file["有期徒刑"] >= 6]

    return final_file


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Shandong sentencing data.")
    parser.add_argument(
        "data_format",
        type=str,
        default="alpaca",
        help="Path to the feature file.",
    )
    parser.add_argument(
        "--rag",
        type=int,
        default=0,
        help="Whether to use RAG data. 0 for no RAG, not 0 for using RAG.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open("rag.txt") as f:
        rag = f.read().strip() + "\n\n"

    final_file = preprocess_xlsx()
    l = format_data(
        final_file,
        rag="" if args.rag == 0 else rag,
        data_format=args.data_format,
    )

    # deterministic shuffle
    random.seed(0)
    random.shuffle(l)

    test_ratio = 0.1
    split_index = int(len(l) * (test_ratio))

    # mutually exclusive splits, 8:1:1
    to_json(
        {
            "validation": l[:split_index],
            "test": l[split_index : split_index * 2],
            "train": l[split_index * 2 :],
        }
    )
