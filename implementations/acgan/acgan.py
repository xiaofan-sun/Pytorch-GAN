from torch.utils.data import DataLoader, Dataset
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse

os.makedirs("result", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1024, help="size of the batches")
parser.add_argument("--discriminator_steps", type=int, default=1, help="Number of discriminator updates to do for each generator update.")
parser.add_argument("--n_classes", type=int, default=2, help="number of classes for dataset")
parser.add_argument("--lr", type=float, default=0.002, help="adam: learning rate")
parser.add_argument("--num_samples", type=int, default=10, help="The number of samples to generate")

parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input z to the generator.')
parser.add_argument('--generator_dim', type=str, default=256, help='Dimension of each generator layer. ')
parser.add_argument('--discriminator_dim', type=str, default=256, help='Dimension of each discriminator layer. ')

parser.add_argument('--train_data', default="../../data/train_data.csv", type=str, help='Path to train data')
parser.add_argument('--output_data', default="../../result/sample_data.csv", type=str, help='Path of the output file')
parser.add_argument('--test_data', default="../../data/test_data.csv", type=str, help='Path to testing data')
parser.add_argument('--output_label', default="../../result/test_labels.csv", type=str, help='Path of the output file')
parser.add_argument('--g_pth', default="../../data/generator.pth", type=str, help='Path to save generator')
parser.add_argument('--d_pth', default="../../result/discriminator.pth", type=str, help='Path of save discriminator')

args = parser.parse_args()
print(args)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        attributes = row[:-1].values.astype(np.float32)
        label = row['label']
        return attributes, label

    def __len__(self):
        return len(self.data)

# 定义生成器网络
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, num_classes, hidden_dim=128):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, input_dim)

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
    
    def fit(self, batch_size, embedding_dim, device):
        z = torch.randn(batch_size, embedding_dim).to(device)
        fake_labels = torch.randint(0, 2, (batch_size,)).to(device)
        fake_samples = self.forward(z, fake_labels)
        return fake_samples, fake_labels

    def forward(self, z, labels):
        gen_input = torch.mul(self.label_emb(labels), z)
        # print("gen_input",gen_input)
        output = self.model(gen_input)
        return output

# 定义判别器网络
class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, input_dim)

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.adv_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.aux_layer = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # print("real_samples",x.size())
        # print("real_labels",labels.size())
        validity = self.model(x)
        adv_output = self.adv_layer(validity)
        aux_output = self.aux_layer(validity)
        return adv_output, aux_output

# def generate_samples(generator, num_samples, hidden_size, device):
#     generator.eval()
#     with torch.no_grad():
#         z = torch.randn(num_samples, hidden_size).to(device)
#         samples = generator(z)
#     generator.train()
#     return samples.cpu().numpy()

def train_acgan(generator, discriminator, dataloader, device, embedding_dim, steps, num_epochs=200, batch_size=64, lr=0.002):
    print("train_acgan")
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)
    
    adv_loss = nn.BCELoss().to(device)
    aux_loss = nn.CrossEntropyLoss().to(device)

    print("start training")
    for epoch in range(num_epochs):
        for i, (attributes, labels) in enumerate(dataloader):
            d_loss = None
            g_loss = None
            for k in range(steps):
                real_samples = attributes.to(device)
                real_labels = labels.to(int).to(device)
                print("real_samples",real_samples)
                # print("real_samples",real_samples.size())
                # print("real_labels",real_labels.size())
                # print("real_labels",real_labels)
                # batch_size = real_samples.size(0)

                # Train discriminator
                optimizer_D.zero_grad()

                fake_samples, fake_labels = generator.fit(batch_size, embedding_dim, device)

                # 判别输出值， 判别标签
                real_adv, real_aux = discriminator(real_samples)
                fake_adv, fake_aux = discriminator(fake_samples)

                # 判别损失
                real_adv_loss = adv_loss(real_adv, torch.ones_like(real_adv))
                fake_adv_loss = adv_loss(fake_adv, torch.zeros_like(fake_adv))
                d_adv_loss = 0.5 * (real_adv_loss + fake_adv_loss)

                # 分类损失
                real_aux_loss = aux_loss(real_aux, real_labels)
                fake_aux_loss = aux_loss(fake_aux, fake_labels)
                d_aux_loss = 0.5 * (real_aux_loss + fake_aux_loss)

                d_loss = d_adv_loss + d_aux_loss
                d_loss.backward()
                optimizer_D.step()

            # Train generator
            optimizer_G.zero_grad()
            fake_samples, fake_labels = generator.fit(batch_size, embedding_dim, device)
            fake_adv, fake_aux = discriminator(fake_samples)
            # 判别损失：
            fake_adv_loss = adv_loss(fake_adv, torch.ones_like(fake_adv))
            # 分类损失
            fake_aux_loss = aux_loss(fake_aux, fake_labels)
            g_loss = fake_adv_loss + fake_aux_loss
            g_loss.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Generator Loss: {g_loss.item():.4f}, Discriminator Loss: {d_loss.item():.4f}")
    print("Training completed!")
    return generator, discriminator

# 加载数据
dataset = MyDataset(args.train_data)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
# 创建模型
generator = Generator(args.embedding_dim, len(dataset[0][0]), args.n_classes, args.generator_dim).to(device)
discriminator = Discriminator(len(dataset[0][0]), args.n_classes, args.discriminator_dim).to(device)
generator, discriminator = train_acgan(generator, discriminator, dataloader, device, args.embedding_dim, args.discriminator_steps, args.num_epochs, args.batch_size, args.lr)
torch.save(generator.state_dict(), args.g_pth)
torch.save(discriminator.state_dict(), args.d_pth)

# 生成样本数据
generated_samples, generated_labels = generator.fit(args.num_samples, args.embedding_dim, device)
generated_data = torch.cat((generated_samples, generated_labels.unsqueeze(1)), dim=1)
df = pd.DataFrame(generated_data.detach().numpy())
df.to_csv(args.output_data, index=False)

# 将测试数据输入到判别器

test_data = pd.read_csv(args.test_data)
test_samples = torch.tensor(test_data.iloc[:, :-1].values.astype(np.float32))
test_labels = torch.tensor(test_data.iloc[:, -1].values)
_, pred_labels = discriminator(test_samples)
pred_labels = torch.argmax(pred_labels.detach(), dim=1)

df = pd.DataFrame(pred_labels)
df.to_csv(args.output_label, index=False)
result = torch.zeros_like(test_labels)
result[pred_labels == test_labels] = 1
print(result)