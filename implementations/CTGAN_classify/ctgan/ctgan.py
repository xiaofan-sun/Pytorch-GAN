"""CTGAN module."""

import warnings

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional, Softmax
from tqdm import tqdm
from base import BaseSynthesizer, random_state

from data_sampler import DataSampler
from data_transformer import DataTransformer

class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=1):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 2)]  # Modify the output layer to have 2 classes
        self.seq = Sequential(*seq)
        self.softmax = Softmax(dim=1)  # Softmax activation for classification

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=1, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        output = self.seq(input_.view(-1, self.pacdim))
        output = self.softmax(output)  # Apply softmax activation
        return output
    
class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)
    
class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data


class CTGAN(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=1, cuda=True):

        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        # if not cuda or not torch.cuda.is_available():
        #     device = 'cpu'
        # elif isinstance(cuda, str):
        #     device = cuda
        # else:
        #     device = 'cuda'
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("device:",device)

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self._discriminator = None
        self.loss_values = None

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        print("Fit the CTGAN Synthesizer models to the training data.")
        # 验证离散列
        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        # 根据数据的特征（连续或离散）来拟合相应的转换器，将原始训练数据转换为模型可用的格式
        print("create DataTransformer")
        self._transformer = DataTransformer()
        # print("len(train_data):",len(train_data.iloc[0]))
        self._transformer.fit(train_data, discrete_columns)
        print("convert train_data to Transformer")
        # print("len(train_data):",len(train_data.iloc[0]))
        train_data = self._transformer.transform(train_data)
        # print("len(train_data):",len(train_data[0]))
        
        print("create DataSampler")
        # 从训练数据中采样生成假数据
        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        data_dim = self._transformer.output_dimensions
        
        print("create Generator")
        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim
        ).to(self._device)

        print("create Discriminator")
        # print("data_dim:",data_dim)
        # print("self._data_sampler.dim_cond_vec():",self._data_sampler.dim_cond_vec())
        self._discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)

        print("optimize")
        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            self._discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            # epoch_iterator.set_description(description.format(gen=0, dis=0))
            epoch_iterator.set_description(description.format(gen=0.0, dis=0.0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in epoch_iterator:
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    # 从高斯分布中生成随机噪声
                    fakez = torch.normal(mean=mean, std=std)

                    # 从条件向量采样器中获取条件向量
                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        # 没有条件向量，从训练数据中采样实际数据
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col, opt)
                    else:
                        # 有条件向量，将条件向量转换为张量，并将其与随机噪声拼接
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)
                        perm = np.arange(self._batch_size) # 一个随机排列的索引数组，用于对训练数据的列进行重新排序
                        np.random.shuffle(perm)
                        # 通过 DataSampler 从训练数据中采样实际数据，并根据之前的随机排列获取相应的条件向量。
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]

                    # 通过生成器生成假样本，并将其输入到激活函数中进行激活
                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    # 将实际数据转换为张量
                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    # 根据条件向量是否存在，将假样本和实际样本与条件向量进行拼接
                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                        # print("real.size()", real.size())
                        # print("c2.size()", c2.size())
                        # print("real_cat.size()", real_cat.size())
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    # 将拼接后的假样本和实际样本输入到判别器中，获取判别器对假样本和实际样本的输出。
                    y_fake = np.argmax(self._discriminator(fake_cat), axis=1)
                    # y_real = np.argmax(self._discriminator(real_cat), axis=1)
                    y_real = np.argmax(self._discriminator(real_cat), axis=1)

                    # 计算梯度惩罚和判别器的损失
                    pen = self._discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    # 对判别器的梯度进行反向传播和优化
                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                # 再次生成随机噪声和条件向量
                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                # 判断是否需要处理条件向量
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                # 通过生成器生成假样本，并将其输入到激活函数中进行激活
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                # 根据条件向量是否存在，将假样本和条件向量进行拼接，并将拼接后的样本输入到判别器中，获取判别器对假样本的输出
                if c1 is not None:
                    y_fake = np.argmax(self._discriminator(torch.cat([fakeact, c1], dim=1)), axis=1)
                else:
                    y_fake = np.argmax(self._discriminator(fakeact), axis=1)

                # 计算条件损失和生成器的损失
                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                # 对生成器的梯度进行反向传播和优化
                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()

            # 记录生成器损失和判别器损失到 epoch_loss_df 数据帧中
            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss]
            })
            print("epoch:{}  Generator Loss:{}  Discriminator Loss:{}".format(epoch_loss_df['Epoch'].values[0],epoch_loss_df['Generator Loss'].values[0],epoch_loss_df['Discriminator Loss'].values[0]))
            if not self.loss_values.empty:
                self.loss_values = pd.concat(
                    [self.loss_values, epoch_loss_df]
                ).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )
        

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column. 
        在生成合成数据时，通过指定条件列和条件值，使生成的数据中特定条件值出现的概率更高。

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def predict(self, test_data, labels):
        """Predict test_data using the trained discriminator.

        Args:
            test_data (numpy.ndarray or pandas.DataFrame):
                Test data to be discriminatd by the discriminator.
                It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``test_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.

        Returns:
            numpy.ndarray
        """

        print("===========================")
        print("predicting")
        # 将原始测试数据转换为模型可用的格式
        # print("11111len(test_data):",len(test_data.iloc[0]))
        test_data = self._transformer.transform(test_data)
        # print("22222len(test_data):",len(test_data[0]))

        # 将实际数据转换为张量
        test_data = torch.from_numpy(test_data.astype('float32')).to(self._device)

        # 从条件向量采样器中获取条件向量
        condvec = self._data_sampler.sample_condvec(len(test_data))
        if condvec is not None:
            # 有条件向量，将条件向量转换为张量，并将其与随机噪声拼接
            c, _, _, _ = condvec
            c = torch.from_numpy(c).to(self._device)
        else:
            c = None

        # 根据条件向量是否存在，将测试样本与条件向量进行拼接
        if c is not None:
            print("test_data.size()", test_data.size())
            print("c.size()", c.size())
            test_data_cat = torch.cat([test_data, c], dim=1)
        else:
            test_data_cat = test_data

        # 判别输入数据
        print("test_data.size()",test_data_cat.size())
        y_test_data = np.argmax(self._discriminator(test_data_cat), axis=1)
        print("y_test_data.cpu().detach().numpy()",y_test_data.cpu().detach().numpy().reshape(-1))
        result = y_test_data.cpu().detach().numpy().reshape(-1)
        print("labels",labels.values)
        print("result",result)
        result[result>=0.5]=1
        result[result<0.5]=0
        ans = (result == labels).astype(int)
        print("len:",len(ans))
        print("sum:",sum(ans))
        return pd.DataFrame(result)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)