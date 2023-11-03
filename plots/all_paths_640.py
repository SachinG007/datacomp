from pathlib import Path

samples_per_epoch_dict = {
    "laion": 13_100_000,
    "no_filter": 128_000_000,
}

paths = {
    "laion": Path("/project_data2/projects/sachingo/utility_project/mediumscale_laion_5x"),
    "no_filter": Path("/home/sachingo/datacomp/logs/logs/mediumscale_nofilter_5x"),
}

match_with_dict = {
    "laion": "epoch",
    "no_filter": "step",
}

subsample_every_dict = {
    "laion": 3,
    "no_filter": 2,
}
