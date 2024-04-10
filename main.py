
from train import *

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser(description='Script to test DKT.')
    parser.add_argument('--dataset', type=str, default='as12_test', help='')
    parser.add_argument('--hidden_num', type=int, default=512, help='')
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--q_num', type=int, default=720, help='')
    parser.add_argument('--concept_num', type=int, default=102, help='')
    parser.add_argument('--length', type=int, default=300, help='')
    parser.add_argument('--d_model', type=int, default=512, help='')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='')
    parser.add_argument('--dropout', type=float, default=0, help='')
    parser.add_argument('--gpu', type=str, default='1', help='')
    parser.add_argument('--loss_rate', type=float, default='0.25', help='')
    parser.add_argument('--rate1', type=float, default='1', help='')
    parser.add_argument('--rate2', type=float, default='1', help='')

    params = parser.parse_args()
    dataset = params.dataset

    if dataset == 'as12_all':
        params.q_num = 53091
        params.length = 200
        params.concept_num = 265

    experiment(
        dataset = params.dataset,
        hidden_num = params.hidden_num,
        learning_rate = params.learning_rate,
        epochs = params.epochs,
        batch_size = params.batch_size,   
        q_num=params.q_num ,
        concept_num=params.concept_num,
        length=params.length,
        d_model=params.d_model,
        nhead=params.nhead,
        num_encoder_layers=params.num_encoder_layers,
        dropout=params.dropout,
        gpu=params.gpu,
        loss_rate=params.loss_rate,
        rate1=params.rate1,
        rate2=params.rate2,

    )

