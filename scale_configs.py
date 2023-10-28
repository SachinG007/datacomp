SCALE_CONFIGS = {
    "debug": {
        "batch_size": 1024,
        "learning_rate": 1e-4,
        "train_num_samples": 128_000_0,
        "warmup": 500,
        "model": "ViT-B-32",
        "beta2": None,
    },
    "small": {
        "batch_size": 4096,
        "learning_rate": 5e-4,
        "train_num_samples": 12_800_000,
        "warmup": 500,
        "model": "ViT-B-32",
        "beta2": None,
    },
    "small_5x": {
        "batch_size": 4096,
        "learning_rate": 5e-4,
        "train_num_samples": 64_000_000,
        "warmup": 500,
        "model": "ViT-B-32",
        "beta2": None,
    },
    "medium": {
        "batch_size": 4096,
        "learning_rate": 5e-4,
        "train_num_samples": 128_000_000,
        "warmup": 500,
        "model": "ViT-B-32",
        "beta2": None,
    },
    "medium_5x_tmars": {
        "batch_size": 4096,
        "learning_rate": 5e-4,
        "train_num_samples": 633_012_450,
        "warmup": 500,
        "model": "ViT-B-32",
        "beta2": None,
    },
    "medium_5x_ocr": {
        "batch_size": 4096,
        "learning_rate": 5e-4,
        "train_num_samples": 639_214_207,
        "warmup": 500,
        "model": "ViT-B-32",
        "beta2": None,
    },
    "medium_5x": {
        "batch_size": 4096,
        "learning_rate": 5e-4,
        "train_num_samples": 640_000_000,
        "warmup": 500,
        "model": "ViT-B-32",
        "beta2": None,
    },
    "medium_5x_b16": {
        "batch_size": 8192,
        "learning_rate": 5e-4,
        "train_num_samples": 640_000_000,
        "warmup": 500,
        "model": "ViT-B-16",
        "beta2": None,
    },
    "large": {
        "batch_size": 8192,
        "learning_rate": 5e-4,
        "train_num_samples": 1_280_000_000,
        "warmup": 500,
        "model": "ViT-B-16",
        "beta2": None,
    },
    "xlarge": {
        "batch_size": 90112,
        "learning_rate": 1e-3,
        "train_num_samples": 12_800_000_000,
        "warmup": 10000,
        "model": "ViT-L-14",
        "beta2": 0.95,
    },
}

SIMPLE_NAMES = ["debug", "small", "medium", "large", "xlarge"]


def available_scales(simple_names=False):
    if simple_names:
        return SIMPLE_NAMES
    else:
        return sorted(list(SCALE_CONFIGS.keys()))


def get_scale_config(scale):
    if scale not in SCALE_CONFIGS:
        raise ValueError(
            f"Unknown scale: {scale}. Please use one of {available_scales()}"
        )
    return SCALE_CONFIGS[scale]
