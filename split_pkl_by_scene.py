#!/usr/bin/env python3

import sys
import os
import pickle


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    src_pkl = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {src_pkl} ...")
    with open(src_pkl, 'rb') as f:
        data_dict = pickle.load(f)

    scenes = list(data_dict.keys())
    print(f"Found {len(scenes)} scenes: {scenes[:5]} ...")

    for i, scene in enumerate(scenes):
        out_path = os.path.join(out_dir, scene + '.pkl')
        if os.path.isfile(out_path):
            print(f"[{i+1}/{len(scenes)}] {scene}: already exists, skipping")
            continue
        print(f"[{i+1}/{len(scenes)}] {scene}: writing {len(data_dict[scene])} entries ...")
        with open(out_path, 'wb') as f:
            pickle.dump(data_dict[scene], f, protocol=4)
        print(f"  -> {out_path}")

    print("Done.")


if __name__ == '__main__':
    main()
