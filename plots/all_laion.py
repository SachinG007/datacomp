'''
logs/logs/laion-scaling/eval_results_Model-B-16_Data-2B_Samples-13B_lr-1e-3_bs-88k.jsonl
logs/logs/laion-scaling/eval_results_Model-B-16_Data-2B_Samples-34B_lr-1e-3_bs-88k.jsonl
logs/logs/laion-scaling/eval_results_Model-B-16_Data-2B_Samples-3B_lr-1e-3_bs-88k.jsonl
logs/logs/laion-scaling/eval_results_Model-B-16_Data-400M_Samples-13B_lr-5e-4_bs-33k.jsonl
logs/logs/laion-scaling/eval_results_Model-B-16_Data-400M_Samples-34B_lr-1e-3_bs-88k.jsonl
logs/logs/laion-scaling/eval_results_Model-B-16_Data-400M_Samples-3B_lr-1e-3_bs-88k.jsonl
logs/logs/laion-scaling/eval_results_Model-B-16_Data-80M_Samples-13B_lr-1e-3_bs-88k.jsonl
logs/logs/laion-scaling/eval_results_Model-B-16_Data-80M_Samples-34B_lr-1e-3_bs-88k.jsonl
logs/logs/laion-scaling/eval_results_Model-B-16_Data-80M_Samples-3B_lr-1e-3_bs-88k.jsonl
logs/logs/laion-scaling/eval_results_Model-B-32_Data-2B_Samples-13B_lr-5e-4_bs-32k.jsonl
logs/logs/laion-scaling/eval_results_Model-B-32_Data-2B_Samples-34B_lr-1e-3_bs-79k.jsonl
logs/logs/laion-scaling/eval_results_Model-B-32_Data-2B_Samples-3B_lr-1e-3_bs-88k.jsonl
logs/logs/laion-scaling/eval_results_Model-B-32_Data-400M_Samples-13B_lr-1e-3_bs-86k.jsonl
logs/logs/laion-scaling/eval_results_Model-B-32_Data-400M_Samples-34B_lr-5e-4_bs-32k.jsonl
logs/logs/laion-scaling/eval_results_Model-B-32_Data-400M_Samples-3B_lr-1e-3_bs-88k.jsonl
logs/logs/laion-scaling/eval_results_Model-B-32_Data-80M_Samples-13B_lr-5e-4_bs-32k.jsonl
logs/logs/laion-scaling/eval_results_Model-B-32_Data-80M_Samples-34B_lr-1e-3_bs-88k.jsonl
logs/logs/laion-scaling/eval_results_Model-B-32_Data-80M_Samples-3B_lr-5e-4_bs-32k.jsonl
/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-L-14_Data-2B_Samples-13B_lr-1e-3_bs-86k.jsonl
/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-L-14_Data-2B_Samples-34B_lr-1e-3_bs-86k.jsonl
/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-L-14_Data-2B_Samples-3B_lr-1e-3_bs-88k.jsonl
/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-L-14_Data-400M_Samples-13B_lr-1e-3_bs-86k.jsonl
/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-L-14_Data-400M_Samples-34B_lr-1e-3_bs-86k.jsonl
/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-L-14_Data-400M_Samples-3B_lr-1e-3_bs-88k.jsonl
/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-L-14_Data-80M_Samples-13B_lr-1e-3_bs-88k.jsonl
/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-L-14_Data-80M_Samples-34B_lr-1e-3_bs-88k.jsonl
/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-L-14_Data-80M_Samples-3B_lr-1e-3_bs-88k.jsonl'''

samples_per_epoch_dict = {
    "2B": 2_000_000_000,
    "400M": 400_000_000,
    "80M": 80_000_000,
}

paths_b16 = {
    "2B": {
        "13B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-B-16_Data-2B_Samples-13B_lr-1e-3_bs-88k.jsonl",
        "34B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-B-16_Data-2B_Samples-34B_lr-1e-3_bs-88k.jsonl",
        "3B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-B-16_Data-2B_Samples-3B_lr-1e-3_bs-88k.jsonl",
    },
    "400M": {
        "13B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-B-16_Data-400M_Samples-13B_lr-5e-4_bs-33k.jsonl",
        "34B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-B-16_Data-400M_Samples-34B_lr-1e-3_bs-88k.jsonl",
        "3B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-B-16_Data-400M_Samples-3B_lr-1e-3_bs-88k.jsonl",
    },
    "80M": {
        "13B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-B-16_Data-80M_Samples-13B_lr-1e-3_bs-88k.jsonl",
        "34B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-B-16_Data-80M_Samples-34B_lr-1e-3_bs-88k.jsonl",
        "3B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-B-16_Data-80M_Samples-3B_lr-1e-3_bs-88k.jsonl",
    },
}

paths_b32 = {
    "2B": {
        "13B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-B-32_Data-2B_Samples-13B_lr-5e-4_bs-32k.jsonl",
        "34B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-B-32_Data-2B_Samples-34B_lr-1e-3_bs-79k.jsonl",
        "3B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-B-32_Data-2B_Samples-3B_lr-1e-3_bs-88k.jsonl",
    },
    "400M": {
        "13B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-B-32_Data-400M_Samples-13B_lr-1e-3_bs-86k.jsonl",
        "34B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-B-32_Data-400M_Samples-34B_lr-5e-4_bs-32k.jsonl",
        "3B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-B-32_Data-400M_Samples-3B_lr-1e-3_bs-88k.jsonl",
    },
    "80M": {
        "13B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-B-32_Data-80M_Samples-13B_lr-5e-4_bs-32k.jsonl",
        "34B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-B-32_Data-80M_Samples-34B_lr-1e-3_bs-88k.jsonl",
        "3B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-B-32_Data-80M_Samples-3B_lr-5e-4_bs-32k.jsonl",
    },
}

paths_l14 = {
    "2B": {
        "13B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-L-14_Data-2B_Samples-13B_lr-1e-3_bs-86k.jsonl",
        "34B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-L-14_Data-2B_Samples-34B_lr-1e-3_bs-86k.jsonl",
        "3B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-L-14_Data-2B_Samples-3B_lr-1e-3_bs-88k.jsonl",
    },
    "400M": {
        "13B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-L-14_Data-400M_Samples-13B_lr-1e-3_bs-86k.jsonl",
        "34B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-L-14_Data-400M_Samples-34B_lr-1e-3_bs-86k.jsonl",
        "3B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-L-14_Data-400M_Samples-3B_lr-1e-3_bs-88k.jsonl",
    },
    "80M": {
        "13B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-L-14_Data-80M_Samples-13B_lr-1e-3_bs-88k.jsonl",
        "34B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-L-14_Data-80M_Samples-34B_lr-1e-3_bs-88k.jsonl",
        "3B": "/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_Model-L-14_Data-80M_Samples-3B_lr-1e-3_bs-88k.jsonl",
    },
}

subsample_every_dict = {
    "2B": 1,
    "400M": 1,
    "80M": 1,
}

match_with_dict = {
    "2B": "step",
    "400M": "step",
    "80M": "step",
}
