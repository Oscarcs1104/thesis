import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(r"C:\CVAIL\Thesis\test\plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_experiment_name(csv_path):
    """
    Convierte una ruta como:

    C:\CVAIL\Thesis\test\checkpoints\FreeSolv\Gatv2\Concat\best.metrics.csv

    en:

    Gatv2-Concat
    """

    p = Path(csv_path)

    try:
        backbone = p.parts[-3]
        fusion = p.parts[-2]
        return f"{backbone}-{fusion}"
    except Exception:
        return p.stem


def plot_metric(csv_paths, metric, split):

    plt.figure(figsize=(12, 7))

    for csv_path in csv_paths:

        try:
            df = pd.read_csv(csv_path)

            data = (
                df[df["split"] == split]
                .sort_values("epoch")
            )

            if len(data) == 0:
                print(f"[WARNING] No hay datos '{split}' en:")
                print(csv_path)
                continue

            name = get_experiment_name(csv_path)

            plt.plot(
                data["epoch"],
                data[metric],
                linewidth=2,
                label=name
            )

        except Exception as e:
            print(f"[ERROR] No se pudo leer:")
            print(csv_path)
            print(e)

    plt.title(f"{split.capitalize()} {metric.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel(metric.upper())
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{split}_{metric}.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Guardado: {output_path}")


def plot_reports(csv_paths):

    metrics = ["loss", "rmse"]
    splits = ["train", "val"]

    for metric in metrics:
        for split in splits:
            plot_metric(csv_paths, metric, split)


if __name__ == "__main__":

    csv_files = [
        r"C:\CVAIL\Thesis\test\checkpoints\FreeSolv\Gatv2\Concat\best.metrics.csv",
        r"C:\CVAIL\Thesis\test\checkpoints\FreeSolv\Gatv2\Mola\best.metrics.csv",
        r"C:\CVAIL\Thesis\test\checkpoints\FreeSolv\Gatv2\Molprop\best.metrics.csv",
        r"C:\CVAIL\Thesis\test\checkpoints\FreeSolv\Gcn\Concat\best.metrics.csv",
        r"C:\CVAIL\Thesis\test\checkpoints\FreeSolv\Gcn\Mola\best.metrics.csv",
        r"C:\CVAIL\Thesis\test\checkpoints\FreeSolv\Gcn\Molprop\best.metrics.csv",
        r"C:\CVAIL\Thesis\test\checkpoints\FreeSolv\Gin\Concat\best.metrics.csv",
        r"C:\CVAIL\Thesis\test\checkpoints\FreeSolv\Gin\Mola\best.metrics.csv",
        r"C:\CVAIL\Thesis\test\checkpoints\FreeSolv\Gin\Molprop\best.metrics.csv",
    ]

    plot_reports(csv_files)