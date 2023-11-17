from pathlib import Path

samples_per_epoch_dict = {
    # "no_filter_16M": 16_000_000,
    # "no_filter_32M": 32_000_000,
    # "no_filter_64M": 64_000_000,
    # "no_filter_128M": 128_000_000,
    "no_filter": 256_000_000,
}

# paths = {
#         "no_filter_16M": Path("/project_data2/projects/sachingo/datacomp_checkpoints/logs/nofilter_16M"),
#         # "no_filter_32M": Path("/project_data2/projects/sachingo/datacomp_checkpoints/logs/nofilter_32M"),
#         # "no_filter_64M": Path("/project_data2/projects/sachingo/datacomp_checkpoints/logs/nofilter_64M"),
#         # "no_filter_128M": Path("/project_data2/projects/sachingo/utility_project/mediumscale_nofilter"),
#         }

paths_16 = {
        "no_filter": Path("/project_data2/projects/sachingo/datacomp_checkpoints/logs/nofilter_16M"),
}

paths_32 = {
        "no_filter": Path("/project_data2/projects/sachingo/datacomp_checkpoints/logs/nofilter_32M"),
}

paths_64 = {
        "no_filter": Path("/project_data2/projects/sachingo/datacomp_checkpoints/logs/nofilter_64M"),
}

paths_128 = {
        "no_filter": Path("/project_data2/projects/sachingo/utility_project/mediumscale_nofilter"),
}

match_with_dict = {
    "no_filter": 'step',
    # "no_filter": 'last_step',
    # "no_filter": 'last_step',
    # "no_filter_": 'last_step',
}

subsample_every_dict = {
    "no_filter": 1,
    "no_filter": 1,
    "no_filter": 1,
    "no_filter": 1,
}
