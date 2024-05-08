from torch.utils.data import DataLoader, Dataset
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import precision_score, accuracy_score, recall_score
import torch.autograd as autograd

os.makedirs("result", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1024, help="size of the batches")
parser.add_argument("--discriminator_steps", type=int, default=1, help="Number of discriminator updates to do for each generator update.")
parser.add_argument("--n_classes", type=int, default=2, help="number of classes for dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--num_samples", type=int, default=10, help="The number of samples to generate")

parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input z to the generator.')
parser.add_argument('--generator_dim', type=str, default=256, help='Dimension of each generator layer. ')
parser.add_argument('--discriminator_dim', type=str, default=256, help='Dimension of each discriminator layer. ')

parser.add_argument('--train_data', default="../../data/train_data.csv", type=str, help='Path to train data')
parser.add_argument('--output_data', default="../../result/sample_data.csv", type=str, help='Path of the output file')
parser.add_argument('--test_data', default="../../data/test_data.csv", type=str, help='Path to testing data')
parser.add_argument('--output_label', default="../../result/test_labels.csv", type=str, help='Path of the output file')
parser.add_argument('--g_pth', default="../../result/generator.pth", type=str, help='Path to save generator')
parser.add_argument('--d_pth', default="../../result/discriminator.pth", type=str, help='Path of save discriminator')
parser.add_argument('--lambda_gp', default=5, type=int)

args = parser.parse_args()
print(args)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 定义生成器网络
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, output_dim),
            # nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# 定义判别器网络
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print("real_samples",x.size())
        validity = self.model(x)
        return validity

def gradient_penalty(discriminator, real_samples, fake_samples):
    # print("real_samples",real_samples.size())
    # print("fake_samples",fake_samples.size())
    # 通过将alpha与真实样本和生成样本进行线性组合，得到介于真实样本和生成样本之间的插值样本,被标记为需要计算梯度。
    alpha = torch.rand(real_samples.size(0), 1)
    alpha = alpha.expand(real_samples.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    # 得到判别器对插值样本的输出
    d_interpolates = discriminator(interpolates)
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(d_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_acgan(generator, discriminator, dataset, device, embedding_dim, steps, num_epochs=200, batch_size=64, lr=0.0002, lambda_gp=10):
    print("train_acgan")
    adv_loss = nn.BCELoss().to(device)
    
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)
    
    print("start training")
    for epoch in range(num_epochs):
        d_loss = None
        g_loss = None
        for k in range(steps):
            real_samples = dataset.to(device)

            # Train discriminator
            optimizer_D.zero_grad()

            z = torch.randn(batch_size, embedding_dim).to(device)
            fake_samples = generator(z)

            real_adv = discriminator(real_samples)
            fake_adv = discriminator(fake_samples)
            # print("real_adv",torch.flatten(real_adv))
            # print("fake_adv",torch.flatten(fake_adv))

            # gp = gradient_penalty(discriminator, real_samples, fake_samples)
            # d_loss = -(torch.mean(real_adv) - torch.mean(fake_adv)) + lambda_gp * gp
            real_adv_loss = adv_loss(real_adv, torch.ones_like(real_adv))
            fake_adv_loss = adv_loss(fake_adv, torch.zeros_like(fake_adv))
            d_loss = 0.5 * (real_adv_loss + fake_adv_loss)

            d_loss.backward()
            optimizer_D.step()

        # Train generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, embedding_dim).to(device)
        fake_samples = generator(z)
        fake_adv = discriminator(fake_samples)
        # g_loss = -torch.mean(fake_adv)
        g_loss = adv_loss(fake_adv, torch.ones_like(fake_adv))
        g_loss.backward()
        optimizer_G.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Generator Loss: {g_loss.item():.4f}, Discriminator Loss: {d_loss.item():.4f}")
    print("Training completed!")
    return generator, discriminator

dataset = torch.tensor(pd.read_csv(args.train_data).values.astype(np.float32))

# 创建模型
generator = Generator(args.embedding_dim, len(dataset[0]), args.generator_dim).to(device)
discriminator = Discriminator(len(dataset[0]), args.discriminator_dim).to(device)
generator, discriminator = train_acgan(generator, discriminator, dataset, device, args.embedding_dim, args.discriminator_steps, args.num_epochs, len(dataset), args.lr, args.lambda_gp)
torch.save(generator.state_dict(), args.g_pth)
torch.save(discriminator.state_dict(), args.d_pth)

# 生成样本数据
z = torch.randn(args.num_samples, args.embedding_dim).to(device)
generated_samples = generator(z)
df = pd.DataFrame(generated_samples.detach().numpy())
df.to_csv(args.output_data, index=False)
# print("generated_data",generated_samples)

# 将测试数据输入到判别器
test_data = pd.read_csv(args.test_data)
# print("test_data",test_data)
test_samples = torch.tensor(test_data.iloc[:, :-1].values.astype(np.float32))
true_labels = torch.tensor(test_data.iloc[:, -1].values)
tem = discriminator(test_samples)
print("pred_res",torch.flatten(tem))
pred_labels = torch.where( tem>=0.9, torch.tensor(1), torch.tensor(0))
# print("pred_labels",torch.flatten(pred_labels))
df = pd.DataFrame(pred_labels)
df.to_csv(args.output_label, index=False)
result = torch.zeros_like(true_labels)
result[torch.flatten(pred_labels) == true_labels] = 1
precision = precision_score(true_labels,pred_labels)
acc = accuracy_score(true_labels,pred_labels)
recall = recall_score(true_labels,pred_labels)
# print("result:",result)
# print("pred_labels:",pred_labels[:30])
# print("true_labels:",true_labels[:30])
print("true_labels_1:",sum(true_labels))
print("pred_labels_1:",sum(pred_labels))
print("total:",len(true_labels))
print("result:",sum(result))
print("acc:",acc)
print("recall:",recall)
print("precision:",precision)