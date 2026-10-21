import argparse
import math

from pmir.config import (
    BATCH_SIZE,
    EPOCHS,
    FEATURE_CH,
    IMG_SIZE,
    LR,
    NUM_WORKERS,
    SAVE_INTERVAL,
    USE_AUGMENT,
    USE_RETINEX,
    WEIGHT_DECAY,
)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Expected a nonnegative integer.")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Expected a finite positive number.")
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Expected a finite nonnegative number.")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train PMIR on paired polarization images.")
    parser.add_argument("--scatter-root", default="data/scatter")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/pmir")
    parser.add_argument("--epochs", type=positive_int, default=EPOCHS)
    parser.add_argument("--batch-size", type=positive_int, default=BATCH_SIZE)
    parser.add_argument("--image-size", type=positive_int, default=IMG_SIZE)
    parser.add_argument("--feature-channels", type=positive_int, default=FEATURE_CH)
    parser.add_argument("--lr", type=positive_float, default=LR)
    parser.add_argument("--weight-decay", type=nonnegative_float, default=WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=nonnegative_int, default=NUM_WORKERS)
    parser.add_argument("--save-interval", type=positive_int, default=SAVE_INTERVAL)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=USE_AUGMENT)
    parser.add_argument("--retinex", action=argparse.BooleanOptionalAction, default=USE_RETINEX)
    args = parser.parse_args(argv)
    if args.image_size < 16 or args.image_size % 8:
        parser.error("--image-size must be a multiple of 8 and at least 16.")
    if args.feature_channels % 2:
        parser.error("--feature-channels must be even.")
    return args


def main(argv=None):
    args = parse_args(argv)
    from pmir.engine.trainer import train

    train(args)


if __name__ == "__main__":
    main()
