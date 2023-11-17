from pathlib import Path

samples_per_epoch_dict = {
    "no_filter_1ep": 128_000_000,
    "clipbucket_top10p": 12_800_000,
    "clipbucket_10p_to_20p": 12_800_000,
    "clipbucket_20p_to_30p": 12_800_000,
    "clipbucket_30p_to_40p": 12_800_000,
    "clipbucket_40p_to_50p": 12_800_000,
    "clipbucket_50p_to_60p": 12_800_000,
    "clipbucket_60p_to_bottom_rand10p": 128_000_00,
    "clipbucket_60p_to_bottom": 12_800_000*4,
    "no_filter_10ep": 12_800_000,
    "clipbucket_top30p_10prandom": 12_800_000,   
}

paths = {
        # "no_filter_1ep": Path("/project_data2/projects/sachingo/utility_project/mediumscale_nofilter"),
        # "no_filter_10ep": Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/wrong_runs/clipbucket_30p_to_40p/"),
        "clipbucket_top10p": Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_top10p"),
        "clipbucket_10p_to_20p": Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_10p_to_20p"),
        "clipbucket_20p_to_30p": Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_20p_to_30p"),
        "clipbucket_30p_to_40p": Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_30p_to_40p"),
        "clipbucket_40p_to_50p": Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_40p_to_50p"),
        "clipbucket_50p_to_60p": Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_50p_to_60p"),
        # "clipbucket_60p_to_bottom_rand10p": Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_60p_to_bottom_rand10p/"),
        "clipbucket_60p_to_bottom": Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_60p_to_bottom/"),
        # "clipbucket_top30p_10prandom": Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_top30p_10prandom/"),
        }


match_with_dict = {
    "no_filter_1ep": "step",
    "clipbucket_top10p": "step",
    "clipbucket_10p_to_20p": "step",
    "clipbucket_20p_to_30p": "step", 
    "clipbucket_30p_to_40p": "step",
    "clipbucket_40p_to_50p": "step",
    "clipbucket_50p_to_60p": "step",
    "clipbucket_60p_to_bottom_rand10p": "step",
    "clipbucket_60p_to_bottom": "step",
    "clipbucket_top30p_10prandom": "step",
    "no_filter_10ep": "step",
}

subsample_every_dict = {
    "no_filter_1ep": 2,
    "clipbucket_top10p": 2,
    "clipbucket_10p_to_20p": 2,
    "clipbucket_20p_to_30p": 2,
    "clipbucket_30p_to_40p": 2,
    "clipbucket_40p_to_50p": 2,
    "clipbucket_50p_to_60p": 2,
    "clipbucket_60p_to_bottom_rand10p": 2,
    "clipbucket_60p_to_bottom": 2,
    "clipbucket_top30p_10prandom": 2,
    "no_filter_10ep": 2,
}
