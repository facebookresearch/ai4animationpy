import torch
import torch.nn as nn
from ai4animation.AI import Modules, Stats


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super(Model, self).__init__()

        self.XStats = Stats.RunningStats(input_dim)
        self.YStats = Stats.RunningStats(output_dim)

        self.XDim = input_dim
        self.YDim = output_dim

        self.Layers = Modules.LinearEncoder(input_dim, hidden_dim, output_dim, dropout)

    def forward(self, x):
        x = self.XStats.Normalize(x)
        y = self.Layers(x)
        return self.YStats.Denormalize(y)

    def learn(self, input, output, update_stats):
        input = (
            self.XStats.UpdateAndNormalize(input)
            if update_stats
            else self.XStats.Normalize(input)
        )
        output = (
            self.YStats.UpdateAndNormalize(output)
            if update_stats
            else self.YStats.Normalize(output)
        )

        y = self.Layers(input)

        mse_fn = nn.MSELoss()
        pred = self.YStats.Denormalize(y)
        loss = mse_fn(y, output)
        return {"Y": pred}, {"MSE": loss}

    # def save(self, path, onnxModel=True, torchModel=True):
    #     if onnxModel:
    #         print("Saving ONNX model.")
    #         utility.SaveONNX(
    #             path=path+'.onnx',
    #             model=self,
    #             input_size=(
    #                 torch.zeros(1, self.XDim)
    #             ),
    #             input_names=['X'],
    #             output_names=['Y']
    #         )
    #     if torchModel:
    #         print("Saving PyTorch model.")
    #         torch.save(self, path+'.pt')
