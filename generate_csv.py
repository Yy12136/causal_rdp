import csv
import re
from pathlib import Path

# ---------------- 从 collect_data 读取并转换 ----------------

def parse_line(line: str):
    """
    将单行文本解析为：
      {
        "score": float or "",
        "components": [(name, weight_str), ...]
      }
    """
    # 非标准行：尽量尝试当作 score
    if not line.startswith("["):
        try:
            score = float(line)
        except ValueError:
            score = ""
        return {"score": score, "components": []}

    m = re.search(r"\[(.*?)\]", line)
    if not m:
        return None

    inside = m.group(1)
    tokens = [t.strip() for t in inside.split(",")]

    # 最后一个 token 是 score
    try:
        score = float(tokens[-1])
        tokens = tokens[:-1]
    except ValueError:
        score = ""

    components = []
    i = 0
    while i < len(tokens):
        name = tokens[i]
        weight = tokens[i + 1] if i + 1 < len(tokens) else ""
        if name.strip():
            components.append((name, weight))
        i += 2

    return {"score": score, "components": components}


def convert_txt_to_csv(txt_path: Path, data_dir: Path, seed: int = 42) -> Path:
    """
    读取一个任务 txt，并输出同名 CSV：
    - CSV 文件名：<txt_stem>.csv
    - env_id：<txt_stem>
    """
    txt_text = txt_path.read_text(encoding="utf-8")
    lines = [l.strip() for l in txt_text.splitlines() if l.strip()]

    parsed_rows = []
    all_component_names = set()
    for line in lines:
        parsed = parse_line(line)
        if not parsed:
            continue
        parsed_rows.append(parsed)
        for name, _ in parsed["components"]:
            if name.strip():
                all_component_names.add(name)

    component_names = sorted(all_component_names)

    fieldnames = ["score"]
    for name in component_names:
        fieldnames.append(f"r_{name}")
        fieldnames.append(f"w_{name}")
        fieldnames.append(f"active_{name}")
    fieldnames += ["env_id", "seed"]

    out_path = data_dir / f"{txt_path.stem}.csv"
    # 每次转换都删除旧文件，确保重新创建（防止意外追加/残留）
    out_path.unlink(missing_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in parsed_rows:
            out_row = {"score": row["score"]}

            # 先默认都空/0
            for name in component_names:
                out_row[f"r_{name}"] = ""
                out_row[f"w_{name}"] = 0
                out_row[f"active_{name}"] = 0

            # 把这一行真正出现的组件填上
            for name, weight in row["components"]:
                try:
                    out_row[f"w_{name}"] = float(weight)
                except ValueError:
                    out_row[f"w_{name}"] = weight
                out_row[f"active_{name}"] = 1

            out_row["env_id"] = txt_path.stem
            out_row["seed"] = seed
            writer.writerow(out_row)

    print("写好了 ->", out_path)
    return out_path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    collect_dir = project_root / "collect_data"
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(collect_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"在 {collect_dir} 下没找到任何 .txt 文件")

    for txt_path in txt_files:
        convert_txt_to_csv(txt_path, data_dir=data_dir, seed=42)


if __name__ == "__main__":
    main()
