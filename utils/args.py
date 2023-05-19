import argparse

OPTIMIZER = ['SGD', 'Adam']
POLICIES = ['poly', 'step', 'None']
ENV = ['colab', 'other']

def get_parser():
    
    parser = argparse.ArgumentParser(
        prog="MLDL23-FL-Project",
        description="Program to test different models in the context of Semantic Segmentation",
        epilog="Thanks for running me :-)"
    )

    subparser = parser.add_subparsers(title="subcommands", dest="framework", help="framework type")
    
    centralized_parser = subparser.add_parser("centralized", help="Use centralized training for Semantic Segmentation")

    federated_parser = subparser.add_parser("federated", help="Use federated training for Semantic Segmentation")

    # Common options
    general = parser.add_argument_group("Run option")
    general.add_argument('--exp_name', type=str, required=True, help='name of the experiment running')
    general.add_argument('--seed', type=int, default=0, help='random seed')
    general.add_argument('--dataset', type=str, choices=['idda', 'femnist'], required=True, help='dataset name')
    general.add_argument('--model', type=str, choices=['deeplabv3_mobilenetv2', 'resnet18', 'cnn'], help='model name')
    general.add_argument('--num_epochs', type=int, help='number of local epochs')

    # Federated options
    federated_args = federated_parser.add_argument_group("Options for federated experiments")
    federated_args.add_argument('--num_rounds', type=int, help='number of rounds')
    federated_args.add_argument('--clients_per_round', type=int, help='number of clients trained per round')
    federated_args.add_argument('--print_train_interval', type=int, default=10, help='client print train interval')
    federated_args.add_argument('--print_test_interval', type=int, default=10, help='client print test interval')
    federated_args.add_argument('--eval_interval', type=int, default=10, help='eval interval')
    federated_args.add_argument('--test_interval', type=int, default=10, help='test interval')
    
    # Learning options
    learning_args = parser.add_argument_group("Options for learning")
    learning_args.add_argument('--hnm', action='store_true', default=False, help='Use hard negative mining reduction or not')
    learning_args.add_argument('--lr', type=float, default=0.05, help='learning rate')
    learning_args.add_argument('--bs', type=int, default=4, help='batch size')
    learning_args.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    learning_args.add_argument('--momentum', type=float, default=0.9, help='momentum')
    learning_args.add_argument('--optimizer', type=str, default='SGD', choices=OPTIMIZER, help='optimizer type')
    

    # Scheduler options
    scheduler_args = parser.add_argument_group("Options for scheduling")
    scheduler_args.add_argument('--lr_policy', type=str, default='poly', choices=POLICIES, help='lr schedule policy')
    scheduler_args.add_argument('--lr_power', type=float, default=0.9, help='power for polyLR')
    scheduler_args.add_argument('--lr_decay_step', type=int, default=15, help='decay step for stepLR')
    scheduler_args.add_argument('--lr_decay_factor', type=float, default=0.1, help='decay factor for stepLR')

    # Transformer options
    transforms_args = parser.add_argument_group("Options of transforms")
    transforms_args.add_argument('--min_scale', type=float, default=0.25, help='define the lowest value for scale')
    transforms_args.add_argument('--max_scale', type=float, default=2.0, help='define the highest value for scale')
    transforms_args.add_argument('--h_resize', type=int, default=512, help='define the resize value for image H ')
    transforms_args.add_argument('--w_resize', type=int, default=1024, help='define the resize value for image W ')
    transforms_args.add_argument('--use_test_resize', action='store_true', help='whether to use test resize')
    transforms_args.add_argument('--jitter', action='store_true', help='whether to use color jitter')
    transforms_args.add_argument('--cv2_transform', action='store_true', help='whether to use cv2_transforms')
    transforms_args.add_argument('--rrc_transform', action='store_true', help='whether to use random resized crop')
    transforms_args.add_argument('--rsrc_transform', action='store_true', help='whether to use random scale random crop')
    transforms_args.add_argument('--cts_norm', action='store_true', help='whether to use cts normalization otherwise 0.5 for mean and std')
    transforms_args.add_argument('--eros_norm', action='store_true', help='whether to use eros normalization otherwise 0.5 for mean and std')

    # Logging options
    logging_args = parser.add_argument_group("Options for logging")
    logging_args.add_argument('--not_use_wandb', action='store_true', help="disable experiment track with wandb")
    logging_args.add_argument('--not_use_serializer', action='store_true', help="disable experiment track with build-in serializer")
    
    return parser
