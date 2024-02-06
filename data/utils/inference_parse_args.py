# -*- coding: utf-8 -*-
import argparse

def parse_args():
    """argument parser for model inference"""

    parser = argparse.ArgumentParser(description="Run DISNET.")
    # Parameters of I/O
    parser.add_argument('--data_input', nargs='?', default="",
                        help='KP table addresses as inputs of the component.')
    parser.add_argument('--tdw_user', type=str, default="",
                        help='tdw_user.')
    parser.add_argument('--tdw_pwd', type=str, default="",
                        help='tdw_user.')
    parser.add_argument('--task_name', type=str, default="",
                        help='tdw_user.')
    parser.add_argument('--data_output', nargs='?', default="",
                        help='Model output address in KP.')
    parser.add_argument('--L2_regularization', type=float, default=0.0,
                        help='Whether to add L2 regularization.')
    parser.add_argument('--tb_log_dir', nargs='?', default="",
                        help='Tensorboard log dir.')
    parser.add_argument('--test', type=bool, default=False,
                        help='Whether to test the model on a different dataset.')
    parser.add_argument('--model_version', type=str, default="v1",
                        help='Which model version to use')
    parser.add_argument('--model_load', type=bool, default=False,
                        help='Whether to build the model from an existing one.')
    parser.add_argument('--validation_output', type=bool, default=False,
                        help='Whether to output the validation result.')
    parser.add_argument('--test_output', type=bool, default=False,
                        help='Whether to output the test result.')
    parser.add_argument('--skin_model_path', type=str, default="hdfs://ss-ieg-dm-v2/data/turing/kp/smoba_shiyan_21199/skinmodel/Epoch_20/Epoch_20/",
                        help='Model Path.')
    parser.add_argument('--hero_model_path', type=str, default="hdfs://ss-ieg-dm-v2/data/turing/kp/smoba_shiyan_21199/heromodel/Epoch_18/",
                        help='Hero Model Path.')
    parser.add_argument('--skin_feat_path', type=str, default="/skin_feature_with_embedding_512.csv",
                        help='skin_feat_path.')
    parser.add_argument('--hero_feat_path', type=str, default="/hero_feature_0424.csv",
                        help='hero feature path.')
    parser.add_argument('--first_items', type=str,
                        default="130,137,175,193,505,10802,11203,11604,13502,14901,15001,19002,19302",
                        help='first_items of train.')
    parser.add_argument('--first_items_test', type=str,
                        default="110,190,198,11203,11604,12602,13502,14203,15001,19002,19302,19802",
                        help='first_items of test.')
    parser.add_argument('--inference_data_path', type=str, help='tdw inference_data_path.')
    parser.add_argument('--skin_predictions_path', type=str,
                        default="hy_db_dc_dm::smoba_secretshop_season24_result_serenazhu_test", help='skin result output_data_path.')
    parser.add_argument('--hero_predictions_path', type=str,
                        default="hy_db_dc_dm::smoba_secretshop_season24_result_serenazhu_test2",
                        help='hero result output_data_path.')

    # Parameters of optimization and training
    parser.add_argument('--partition_num', type=int, default=5,
                        help='partition_num.')
    parser.add_argument('--save_partition_num', type=int, default=2048,
                        help='save_partition_num.')
    parser.add_argument('--prefetch_size', type=int, default=8,
                        help='Prefetch Size.')
    parser.add_argument('--inference_batch_size', type=int, default=128,
                        help='inference batch size.')

    parser.add_argument('--thresholds', type=float, default=0.5,
                        help='0 1 thresholds.')
    parser.add_argument('--neg_threshold', type=float, default=0.3,
                        help='negative sample select thresholds.')
    parser.add_argument('--decay_rate', type=float, default=0.9,
                        help='Change decay rate of exponential decay learning rate.')

    # Parameters of data loading
    parser.add_argument('--cache_dir', type=str, default="/dockerdata/cache/data",
                        help='cache_dir')
    parser.add_argument('--num_parallel_reads', type=int, default=64,
                        help='Parameters in dateset map.')
    parser.add_argument('--num_parallel_map', type=int, default=64,
                        help='Parameters in dateset map.')
    parser.add_argument('--read_buffer_size', type=int, default=52428800,
                        help='Parameters in TextLineDataset.')
    parser.add_argument('--shuffle_buffer_size', type=int, default=16,
                        help='Parameters in dateset shuffle.')
    parser.add_argument('--is_cache', type=int, default=1, help="cache or not")

    # Parameters of model
    parser.add_argument('--userlayers', nargs='?', default='[128, 64, 32]',
                        help="Size of each user layer")
    parser.add_argument('--itemlayers', nargs='?', default='[3]',
                        help="Size of each item layer")
    parser.add_argument('--string_index', nargs='?', default='[10,11,12,20,22,23]',
                        help="string_index of chosen features")

    args, _ = parser.parse_known_args()

    return args