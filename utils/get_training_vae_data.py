import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser(
        description="Merge GF2/QB/WV3 HRMS gt into a single 1-channel H5 dataset"
    )
    ap.add_argument(
        "--gf2",
        type=str,
        default="data/gf2/train_gf2.h5",
        help="GF-2 train H5 path",
    )
    ap.add_argument(
        "--qb",
        type=str,
        default="data/qb/train_qb.h5",
        help="QuickBird train H5 path",
    )
    ap.add_argument(
        "--wv3",
        type=str,
        default="data/wv3/train_wv3.h5",
        help="WorldView-3 train H5 path",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="data/vae/train_gt_1ch_all.h5",
        help="Output H5 path",
    )
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle sample order before writing",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for shuffling",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    in_infos = []
    # sensor_name, path
    inputs = [
        ("gf2", args.gf2),
        ("qb",  args.qb),
        ("wv3", args.wv3),
    ]

    print("Inspecting input files...")
    for sensor_name, path in inputs:
        if not Path(path).exists():
            print(f"[WARN] Skip sensor={sensor_name}, file not found: {path}")
            continue
        with h5py.File(path, "r") as f:
            if "gt" not in f:
                raise KeyError(f"{path} does not contain dataset 'gt'")
            ds = f["gt"]
            if ds.ndim != 4:
                raise ValueError(f"{path}:/gt expected 4D (N,C,H,W), got {ds.shape}")
            N, C, H, W = ds.shape
            print(f"  {sensor_name}: {path}")
            print(f"    gt shape = {ds.shape}, dtype={ds.dtype}")
            in_infos.append(
                {
                    "sensor": sensor_name,
                    "path": path,
                    "N": N,
                    "C": C,
                    "H": H,
                    "W": W,
                }
            )

    if not in_infos:
        raise RuntimeError("No valid input H5 found. Please check paths.")

    H0 = in_infos[0]["H"]
    W0 = in_infos[0]["W"]
    for info in in_infos:
        if info["H"] != H0 or info["W"] != W0:
            raise ValueError(
                f"All gt datasets must have same H,W. "
                f"Got {info['sensor']}:{info['H']}x{info['W']} vs first {H0}x{W0}"
            )

    sensor_to_id = {info["sensor"]: i for i, info in enumerate(in_infos)}

    sample_index = []  # list of (sensor_id, img_idx, band_idx)
    for s_id, info in enumerate(in_infos):
        N, C = info["N"], info["C"]
        for n in range(N):
            for c in range(C):
                sample_index.append((s_id, n, c))

    total_samples = len(sample_index)
    print(f"Total 1-channel samples = {total_samples}")

    if args.shuffle:
        print(f"Shuffling with seed {args.seed}...")
        rng = np.random.default_rng(args.seed)
        rng.shuffle(sample_index)

    out_path = Path(args.out)
    if out_path.exists():
        print(f"[WARN] Output file already exists, will overwrite: {out_path}")
        out_path.unlink()

    print(f"Creating output H5 file: {out_path}")
    with h5py.File(out_path, "w") as fout:
        gt_ds = fout.create_dataset(
            "gt",
            shape=(total_samples, 1, H0, W0),
            dtype="float64",
            chunks=None,         # contiguous layout
            compression=None,
        )

        sensor_id_ds = fout.create_dataset(
            "sensor_id",
            shape=(total_samples,),
            dtype="int8",
            chunks=None,
            compression=None,
        )
        img_idx_ds = fout.create_dataset(
            "img_index",
            shape=(total_samples,),
            dtype="int64",
            chunks=None,
            compression=None,
        )
        band_idx_ds = fout.create_dataset(
            "band_index",
            shape=(total_samples,),
            dtype="int64",
            chunks=None,
            compression=None,
        )

        sensor_name_ds = fout.create_dataset(
            "sensor_name",
            shape=(len(in_infos),),
            dtype="S8",
            chunks=None,
            compression=None,
        )
        for info in in_infos:
            sid = sensor_to_id[info["sensor"]]
            sensor_name_ds[sid] = info["sensor"].encode("ascii")

        in_files = [h5py.File(info["path"], "r") for info in in_infos]
        try:
            print("Merging data (this may take a while)...")
            for out_idx, (s_id, img_idx, band_idx) in enumerate(
                tqdm(sample_index, total=total_samples)
            ):
                f_in = in_files[s_id]
                gt_in = f_in["gt"]  # (N,C,H,W)

                band = gt_in[img_idx, band_idx, :, :]  # np.ndarray, float64
                gt_ds[out_idx, 0, :, :] = band
                sensor_id_ds[out_idx] = s_id
                img_idx_ds[out_idx] = img_idx
                band_idx_ds[out_idx] = band_idx
        finally:
            for f in in_files:
                f.close()

    print("Done. New merged file saved at:", out_path)
    print("Datasets inside:")
    print("  /gt         : (N,1,H,W) float64")
    print("  /sensor_id  : (N,) int8  (0=gf2,1=qb,2=wv3)")
    print("  /img_index  : (N,) int64 (original image index)")
    print("  /band_index : (N,) int64 (channel index within that image)")
    print("  /sensor_name: (3,) 'S8'  (id -> name mapping)")
    

if __name__ == "__main__":
    main()
