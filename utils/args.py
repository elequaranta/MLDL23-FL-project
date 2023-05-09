import argparse

OPTIMIZER = ['SGD', 'Adam']
POLICIES = ['poly', 'step', 'None']

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True, help='name of the experiment running')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dataset', type=str, choices=['idda', 'femnist'], required=True, help='dataset name')
    parser.add_argument('--centralized', type=bool, default=False, required=True, help='choose between centralized training or federeted training')
    parser.add_argument('--niid', action='store_true', default=False,
                        help='Run the experiment with the non-IID partition (IID by default). Only on FEMNIST dataset.')
    parser.add_argument('--model', type=str, choices=['deeplabv3_mobilenetv2', 'resnet18', 'cnn'], help='model name')
    parser.add_argument('--num_rounds', type=int, help='number of rounds')
    parser.add_argument('--num_epochs', type=int, help='number of local epochs')
    parser.add_argument('--clients_per_round', type=int, help='number of clients trained per round')
    parser.add_argument('--hnm', action='store_true', default=False, help='Use hard negative mining reduction or not')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--bs', type=int, default=4, help='batch size')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--print_train_interval', type=int, default=10, help='client print train interval')
    parser.add_argument('--print_test_interval', type=int, default=10, help='client print test interval')
    parser.add_argument('--eval_interval', type=int, default=10, help='eval interval')
    parser.add_argument('--test_interval', type=int, default=10, help='test interval')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=OPTIMIZER, help='optimizer type')
    
    # Scheduler
    parser.add_argument('--lr_policy', type=str, default='poly', choices=POLICIES, help='lr schedule policy')
    parser.add_argument('--lr_power', type=float, default=0.9, help='power for polyLR')
    parser.add_argument('--lr_decay_step', type=int, default=5000, help='decay step for stepLR')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='decay factor for stepLR')
    
    return parser
