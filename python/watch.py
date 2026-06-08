import argparse
import torch
from visualize import Visualizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Watch a saved Snake AI individual play")
    parser.add_argument("checkpoint", help=".pt file saved by train.py")
    parser.add_argument("--fps",   type=int, default=15)
    parser.add_argument("--grid",  type=int, default=20)
    args = parser.parse_args()

    params = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    print(f"Loaded {args.checkpoint}  ({params.shape[0]} params)  |  device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    viz     = Visualizer(grid_size=args.grid)
    episode = 0

    print("Press Q or close the window to quit.  Each episode uses fresh random food.")
    while True:
        episode += 1
        score = viz.play_top_n([params], DEVICE, generation=episode, fps=args.fps)
        if score == -1:
            break
        print(f"episode {episode:4d}  score {score}")

    viz.close()


if __name__ == "__main__":
    main()
