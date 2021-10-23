import torch
import torch.nn as nn
import numpy as np
from reprod_log import ReprodLogger
from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModeltorch
if __name__ == "__main__":
    model_file = 'model_92_sgd.pkl'
    model_pd_file = 'model_92_sgd.pdparams'
    torch.device("cpu")
    # def logger
    reprod_logger = ReprodLogger()
    model = ResidualAttentionModeltorch()
    model.load_state_dict((torch.load(model_file, map_location='cpu')))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # 读取fake data和fake label
    fake_data = np.load("./fake_data.npy")
    fake_data = torch.from_numpy(fake_data)

    fake_label = np.load("./fake_label.npy")
    fake_label = torch.from_numpy(fake_label)

    # forward
    out = model(fake_data)
    # 计算loss的值
    print(out.shape)
    print(fake_label.shape)
    loss = criterion(out, fake_label)
    # 记录loss到文件中
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("loss_torch.npy")