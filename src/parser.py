import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description='for GNN')
    # model
    parser.add_argument('--node_size', help="Please give a value for node_size")
    parser.add_argument('--seq_len', help="Please give a value for seq_len",
                        default=16, type=int)
    parser.add_argument('--num_layers', help="Please give a value for num_layers",
                        default=6, type=int)
    parser.add_argument('--num_heads', help="Please give a value for num_heads",
                        default=6, type=int)
    parser.add_argument('--hidden_size', help="Please give a value for hidden_size",
                        default=384, type=int)
    parser.add_argument('--intermediate_size', help="Please give a value for intermediate_size",
                        default=1536, type=int)

    parser.add_argument('--attention_dropout', help="Please give a value for attention_dropout",
                        default=0.1, type=float)
    parser.add_argument('--hidden_dropout', help="Please give a value for hidden_dropout",
                        default=0.1, type=float)
    # data
    parser.add_argument('--num_neg', help="Please give a value for num_neg",
                        default=4, type=int)
    # train
    parser.add_argument('--epochs', help="Please give a value for epochs",
                        default=30, type=int)
    parser.add_argument('--batch_size', help="Please give a value for batch_size",
                        default=32, type=int)
    # optimizer
    parser.add_argument('--init_lr', help="Please give a value for init_lr",
                        default=0.0001, type=float)
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor",
                        default=0.5, type=float)
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience",
                        default=100, type=int)
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay",
                        default=0.0005, type=float)
    parser.add_argument('--lr_min', help="Please give a value for min_lr",
                        default=0.000001, type=float)
    parser.add_argument('--warmup_steps', help="Please give a value for warmup_steps",
                        default=20000, type=int)
    # data
    parser.add_argument('--env_path', help="Please give a value for data_path",
                        default='/home/python_projects/ner_playground/', type=str)
    parser.add_argument('--task_inf', help="Please give task info",
                        default='123123', type=str)
    # gpu
    parser.add_argument('--device', help="Please give a value for device",
                        default='cuda', type=str)
    parser.add_argument('--use_gpu', help="Please give a value for use_gpu",
                        default=True, type=bool)
    parser.add_argument('--gpu_id', help="Please give a value for gpu_id",
                        default=1, type=int)

    return parser.parse_args()
