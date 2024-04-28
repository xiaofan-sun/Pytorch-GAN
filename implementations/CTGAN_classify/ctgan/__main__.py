"""CLI."""

import argparse

from data import read_csv, read_tsv, write_tsv, write_tsv_label
from ctgan import CTGAN

def _parse_args():
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        help='Number of training epochs')
    parser.add_argument('-t', '--tsv', action='store_true',
                        help='Load data in TSV format instead of CSV')
    parser.add_argument('--no-header', dest='header', action='store_false',
                        help='The CSV file has no header. Discrete columns will be indices.')

    parser.add_argument('-m', '--metadata', help='Path to the metadata')
    parser.add_argument('-d', '--discrete',
                        help='Comma separated list of discrete columns without whitespaces.')
    parser.add_argument('-n', '--num-samples', type=int,
                        help='Number of rows to sample. Defaults to the training data size')

    parser.add_argument('--generator_lr', type=float, default=2e-4,
                        help='Learning rate for the generator.')
    parser.add_argument('--discriminator_lr', type=float, default=2e-4,
                        help='Learning rate for the discriminator.')

    parser.add_argument('--generator_decay', type=float, default=1e-6,
                        help='Weight decay for the generator.')
    parser.add_argument('--discriminator_decay', type=float, default=0,
                        help='Weight decay for the discriminator.')

    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimension of input z to the generator.')
    parser.add_argument('--generator_dim', type=str, default='256,256',
                        help='Dimension of each generator layer. '
                        'Comma separated integers with no whitespaces.')
    parser.add_argument('--discriminator_dim', type=str, default='256,256',
                        help='Dimension of each discriminator layer. '
                        'Comma separated integers with no whitespaces.')

    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size. Must be an even number.')
    parser.add_argument('--save', default=None, type=str,
                        help='A filename to save the trained synthesizer.')
    parser.add_argument('--load', default=None, type=str,
                        help='A filename to load a trained synthesizer.')

    parser.add_argument('--sample_condition_column', default=None, type=str,
                        help='Select a discrete column name.')
    parser.add_argument('--sample_condition_column_value', default=None, type=str,
                        help='Specify the value of the selected discrete column.')

    parser.add_argument('--data', help='Path to training data')
    parser.add_argument('--output', default="../result/sample_data.csv", type=str,
                        help='Path of the output file')

    parser.add_argument('--test_data', help='Path to testing data')
    parser.add_argument('--output_label', default="../result/label.csv", type=str,
                        help='Path of the output file')

    parser.add_argument('--predict', default=True, type=bool, help='Whether to make a prediction')

    return parser.parse_args()

def main():
    """CLI."""
    args = _parse_args()
    print(args)
    if args.tsv:
        data, discrete_columns = read_tsv(args.data, args.metadata)
        test_data, test_discrete_columns = read_tsv(args.test_data, args.metadata)
        test_data = test_data[:-1]
    else:
        data, discrete_columns = read_csv(args.data, args.metadata, args.header, args.discrete)
        # print("data",data)
        # print("type(data)",type(data)) #type(data) <class 'pandas.core.frame.DataFrame'>
        # print("len(data.iloc[0])",len(data.iloc[0]))
        test_data, test_discrete_columns = read_csv(args.test_data, args.metadata, args.header, args.discrete)
        test_data = test_data[:-1]
        # print("len(test_data.iloc[0])",len(test_data.iloc[0]))

    if args.load:
        print("load CTGAN")
        model = CTGAN.load(args.load)
        
    else:
        print("create CTGAN")
        generator_dim = [int(x) for x in args.generator_dim.split(',')]
        discriminator_dim = [int(x) for x in args.discriminator_dim.split(',')]
        model = CTGAN(
            embedding_dim=args.embedding_dim, generator_dim=generator_dim,
            discriminator_dim=discriminator_dim, generator_lr=args.generator_lr,
            generator_decay=args.generator_decay, discriminator_lr=args.discriminator_lr,
            discriminator_decay=args.discriminator_decay, batch_size=args.batch_size,
            epochs=args.epochs)

        # 将数据和离散列作为参数，对CTGAN模型进行训练
        model.fit(data, discrete_columns)
        # 保存模型
        if args.save is not None:
            model.save(args.save)

    num_samples = args.num_samples or len(data)

    # 确保在使用model.sample()方法生成样本之前，必须同时或都不提供条件列和条件值
    if args.sample_condition_column is not None:
        assert args.sample_condition_column_value is not None

    # 生成指定数量的样本，并将结果存储在sampled变量中
    sampled = model.sample(
        num_samples,
        args.sample_condition_column,
        args.sample_condition_column_value)
    if args.tsv:
        write_tsv(sampled, args.metadata, args.output)
    else:
        sampled.to_csv(args.output, index=False)
    
     # 使用判别器进行预测，并将结果存储在sampled变量中
    pre_result = model.predict(test_data, test_discrete_columns)
    if args.tsv:
        write_tsv(pre_result, args.output_label)
    else:
        pre_result.to_csv(args.output_label, index=False)
    

if __name__ == '__main__':
    main()