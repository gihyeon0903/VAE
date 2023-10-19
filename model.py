import torch
import torch.nn as nn

class gaussian_MLP_encoder(nn.Module):
    def __init__(self, dim_z=2, n_hidden=500, keep_prob=0.5):
      super(gaussian_MLP_encoder, self).__init__()

      self.dim_z = dim_z
      self.softplus = nn.Softplus()

      self.fc1 = nn.Sequential(
          nn.Linear(784, n_hidden),
          nn.ELU(),
          nn.Dropout(p=keep_prob)
      )

      self.fc2 = nn.Sequential(
          nn.Linear(n_hidden, n_hidden),
          nn.Tanh(),
          nn.Dropout(p=keep_prob)
      )

      # pytoch vae 예시를 보면 output layer를 2개(mu, std)를 사용하나, 이활석 박사님 tensorflow 코드를 보면 1개의 output layer만을 사용 따라서, dim_z를 2배로 설정해야함.
      self.output_layer = nn.Sequential(
          nn.Linear(n_hidden, dim_z*2),
      )

      # 가중치 initialize(이활 석 박사님 tensorflow 코드를 보면 variance_scaling_initializer를 사용하나 pytorch에서 지원하지않아 Xavier를 사용)
      for m in self.modules():
        if isinstance(m, nn.Linear):
          torch.nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):

      x = self.fc1(x)
      x = self.fc2(x)
      gaussain_params = self.output_layer(x)

      mean   = gaussain_params[:, :self.dim_z]
      stddev = 1e-6 + self.softplus(gaussain_params[:, self.dim_z:]) # 표준편차는 항상 양수여야하 므로 relu 함수에서 (0, 0) 부근을 부드럽게 변형한 함수인 softplus 함수를 통과 한뒤 0이 나오지 않도록 1e-6을 더해줌

      return mean, stddev

class bernoulli_MLP_decoder(nn.Module):
    def __init__(self, dim_z=2, n_hidden=500, keep_prob=0.5):
      super(bernoulli_MLP_decoder, self).__init__()

      self.fc1 = nn.Sequential(
          nn.Linear(dim_z, n_hidden),
          nn.Tanh(),
          nn.Dropout(keep_prob)
      )

      self.fc2 = nn.Sequential(
          nn.Linear(n_hidden, n_hidden),
          nn.ELU(),
          nn.Dropout(keep_prob)
      )

      self.output_layer = nn.Sequential(
          nn.Linear(n_hidden, 784),
          nn.Sigmoid()
      )

      for m in self.modules():
        if isinstance(m, nn.Linear):
          torch.nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):

      x = self.fc1(x)
      x = self.fc2(x)
      out = self.output_layer(x)

      return out

class Variatinal_Auto_Encoder(nn.Module):
  def __init__(self, dim_z=2, n_hidden=500, keep_prob=0.5, device='cpu'):
      super(Variatinal_Auto_Encoder, self).__init__()
      self.device = device
      self.encoder = gaussian_MLP_encoder(dim_z, n_hidden, keep_prob).to(device)
      self.decoder = bernoulli_MLP_decoder(dim_z, n_hidden, keep_prob).to(device)

  def forward(self, x):

    mean, stddev = self.encoder(x)
    # reparameterization trick으로 z(Latent vector) Sampling
    z = mean + stddev * torch.normal(0, 1, (mean.size()[0], 1)).to(self.device)

    y = self.decoder(z)
    # tf.clip_by_value를 이용해 1e-8 ~ 1-1e-8 사이의 값으로 잘라주는 작업을 하나, sigmoid 함수를 통과하는 데 이 작업을 왜 해주는 지 모르겠음.
    return y, (mean, stddev), z


class Auto_Encoder(nn.Module):
  def __init__(self, dim_z=2, n_hidden=500, keep_prob=0.5):
      super(Auto_Encoder, self).__init__()
      
      self.encoder = nn.Sequential(
          nn.Linear(784, n_hidden),
          nn.ELU(),
          nn.Dropout(p=keep_prob),
          nn.Linear(n_hidden, n_hidden),
          nn.Tanh(),
          nn.Dropout(p=keep_prob),
          nn.Linear(n_hidden, dim_z)
      )
      self.decoder = nn.Sequential(
          nn.Linear(dim_z, n_hidden),
          nn.Tanh(),
          nn.Dropout(keep_prob),
          nn.Linear(n_hidden, n_hidden),
          nn.ELU(),
          nn.Dropout(keep_prob),
          nn.Linear(n_hidden, 784),
          nn.Sigmoid()
      )

      for m in self.modules():
        if isinstance(m, nn.Linear):
          torch.nn.init.xavier_normal_(m.weight.data)

  def forward(self, x):
    z = self.encoder(x)
    out = self.decoder(z)
    return out, z

    