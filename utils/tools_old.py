import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from datetime import datetime
from distutils.util import strtobool
import pandas as pd

from utils.metrics import metric, SampleMAE, SampleMSE
import torch.fft as fft
from einops import rearrange, reduce, repeat
import math
import os

plt.switch_backend('agg')

def adjust_learning_rate(optimizer, scheduler, epoch, args):
    if args.lradj =='type1':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj =='type2':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch - 1) // 1))}
    elif args.lradj =='type4':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch) // 1))}
    elif args.lradj =='TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    else:
        #args.learning_rate = 1e-4
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}

    #print("lr_adjust = {}".format(lr_adjust))
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        #print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        print("保存模型----------")
        print(path + '/' + 'checkpoint.pth')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        print("保存完模型--------")
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def visual_decomp(data, trend, sea, res, name='./pic/test.pdf'):
    """
    Results visualization decompostion
    """
    plt.figure()
    plt.plot(data, label='GroundTruth', linewidth=2)
    plt.plot(trend, label='Trend', linewidth=2)
    plt.plot(sea, label='Seasonality', linewidth=2)
    plt.plot(res, label='Res', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')



def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )




class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean


class FourierLayer(nn.Module):

    def __init__(self, pred_len, k=None, low_freq=1, output_attention=False):
        super().__init__()
        # self.d_model = d_model
        self.pred_len = pred_len
        self.k = k
        self.low_freq = low_freq
        self.output_attention = output_attention

    def forward(self, x):
        """x: (b, t, d)"""

        b, t, d = x.shape
        x_freq = fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = fft.rfftfreq(t)[self.low_freq:-1]
            # print("-------------------")
            # print(x_freq.shape)
            # print(f.shape)
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)

        # print("**********************************")
        # print(x_freq[0,:,-1])
        # print(x_freq[0,:,-2])
        # print(index_tuple[1].shape)
        # print(index_tuple[1][0,:,-1])
        # print(index_tuple[1][0, :, -2])

        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2))
        f = f.to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)


        return self.extrapolate(x_freq, f, t), None

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t_val = rearrange(torch.arange(t + self.pred_len, dtype=torch.float),
                          't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs() / t, 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')

        ##可以取f、amp、phase作为模型的输入，三个维度都是【batch_size, topk, n_vars]


        x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)

        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        values, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)))
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]

        return x_freq, index_tuple



def frequency_extract(x, k, low_freq=1):
    b, t, d = x.shape
    x_freq = fft.rfft(x, dim=1)

    if t % 2 == 0:
        x_freq = x_freq[:, low_freq:-1]
        f = fft.rfftfreq(t)[low_freq:-1]
    else:
        x_freq = x_freq[:, low_freq:]
        f = fft.rfftfreq(t)[low_freq:]

    def topk_freq(x_freq, k):
        values, indices = torch.topk(x_freq.abs(), k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)))
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple

    x_freq, index_tuple = topk_freq(x_freq, k)

    f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2))
    f = f.to(x_freq.device)
    f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)

    # x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
    # f = torch.cat([f, -f], dim=1)

    amp = rearrange(x_freq.abs() / t, 'b f d -> b f () d')
    phase = rearrange(x_freq.angle(), 'b f d -> b f () d')

    ##排序
    f, _ = torch.sort(f, dim=1)
    amp, _ = torch.sort(amp, dim=1)
    phase, _ = torch.sort(phase, dim=1)

    ##可以取f、amp、phase作为模型的输入，三个维度都是【batch_size, topk, n_vars]

    return f, amp, phase




class FourierLayer(nn.Module):

    def __init__(self, pred_len, k=None, low_freq=1, output_attention=False):
        super().__init__()
        # self.d_model = d_model
        self.pred_len = pred_len
        self.k = k
        self.low_freq = low_freq
        self.output_attention = output_attention

    def forward(self, x):
        """x: (b, t, d)"""

        b, t, d = x.shape
        x_freq = fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = fft.rfftfreq(t)[self.low_freq:-1]
            # print("-------------------")
            # print(x_freq.shape)
            # print(f.shape)
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)

        # print("**********************************")
        # print(x_freq[0,:,-1])
        # print(x_freq[0,:,-2])
        # print(index_tuple[1].shape)
        # print(index_tuple[1][0,:,-1])
        # print(index_tuple[1][0, :, -2])

        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2))
        f = f.to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)


        return self.extrapolate(x_freq, f, t), None

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t_val = rearrange(torch.arange(t + self.pred_len, dtype=torch.float),
                          't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs() / t, 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')

        ##可以取f、amp、phase作为模型的输入，三个维度都是【batch_size, topk, n_vars]


        x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)

        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        values, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)))
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]

        return x_freq, index_tuple






class Decomposition(nn.Module):
    def __init__(self, k=3, kernel_size=[4, 8, 12]):
        super().__init__()
        self.seasonality_model = FourierLayer(pred_len=0, k=k)
        self.trend_model = series_decomp_multi(kernel_size)

    def forward(self, x):
        _, trend = self.trend_model(x)
        #seasonality, _ = self.seasonality_model(x-trend)
        #res = x - trend - seasonality
        res = x - trend
        return trend, res




def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

SinCosPosEncoding = PositionalEncoding

def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)








def vali(model, vali_data, vali_loader, criterion, args, device, itr,sampling_rate, flag):
    total_loss = []
    patchtst_loss = []
    ##这里为什么只对in_layer和out_layer进行eval
    # model.in_layer.eval()
    # model.out_layer.eval()
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            output_dic = model(batch_x, batch_y, sampling_rate, flag, test=False)
            outputs = output_dic["outputs"]
            patchtst_outputs = output_dic["patchtst_outputs"]
            #res_outputs = output_dic["res_outputs"]

            ###为了变长的输出
            outputs = outputs[:, :args.pred_len, :]
            patchtst_outputs = patchtst_outputs[:, :args.pred_len, :]


            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            # trend_outputs = output_dic["trend_outputs"].detach().cpu()
            # sea_outputs = output_dic["sea_outputs"].detach().cpu()
            # res_outputs = output_dic["res_outputs"].detach().cpu()
            # trend_truth = output_dic["y_trend"].detach().cpu()
            # res_truth = output_dic["y_res"].detach().cpu()

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            patchtst_pred = patchtst_outputs.detach().cpu()
            # res_pred = res_outputs.detach().cpu()
            #
            # print("--------------------------")
            # print(pred.shape)
            # print(true.shape)

            loss = criterion(pred, true)
            #loss = 0
            time_loss = criterion(patchtst_pred, true)
            # loss = criterion(pred, trend_truth) + criterion(res_pred, res_truth)
            # time_loss = criterion(patchtst_pred, trend_truth) + criterion(res_pred, res_truth)
            total_loss.append(loss)
            patchtst_loss.append(time_loss)
    patchtst_loss = np.average(patchtst_loss)
    total_loss = np.average(total_loss)
    model.train()
    return total_loss, patchtst_loss

def MASE(x, freq, pred, true):
    masep = np.mean(np.abs(x[:, freq:] - x[:, :-freq]))
    return np.mean(np.abs(pred - true) / (masep + 1e-8))


def test(model, test_data, test_loader, args, device, itr, sampling_rate, flag):
    preds = []
    trues = []
    inputx = []
    mae_list = []
    mse_list = []
    CKA_linear_list = []
    CKA_kernel_list = []
    folder_path = './../visualization/ETTm1_720/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            #output_dic = model(batch_x, None, itr)
            output_dic = model(batch_x, batch_y, sampling_rate, flag, test=True)
            outputs = output_dic["outputs"]
            patchtst_outputs = output_dic["patchtst_outputs"]

            cka_kernel_metric = output_dic["cka_kernel_metric"]
            cka_linear_metric = output_dic["cka_linear_metric"]
            CKA_linear_list.append(cka_linear_metric)
            CKA_kernel_list.append(cka_kernel_metric)

            # ###为了变长的输出
            # patchtst_outputs = patchtst_outputs[:, :args.pred_len, :]



            # encoder - decoder
            # outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            pred = patchtst_outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            input = batch_x.detach().cpu().numpy()
            #res_pred = res_outputs.detach().cpu().numpy()
            # mae, mse, rmse, mape, mspe, smape, nd = metric(pred, true)
            # mae_list.append(mae)
            # mse_list.append(mse)

            # if i % 5 == 0:
            #     # input = batch_x.detach().cpu().numpy()
            #     # trend = trend.detach().cpu().numpy()
            #     # #sea = sea.detach().cpu().numpy()
            #     # res = res.detach().cpu().numpy()
            #
            #
            #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
            #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
            #     # visual_decomp(input[0,:,-1], trend[0,:,-1], res[0,:,-1], res[0,:,-1],os.path.join(folder_path, str(i) + 'decomposition.pdf'))
            #     # visual_decomp(true[0, :, -1], pred[0, :, -1], res_pred[0, :, -1], res_pred[0, :, -1],
            #     #               os.path.join(folder_path, str(i) + 'predict_decomposition.pdf'))
            #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

            preds.append(pred)
            trues.append(true)
            inputx.append(input)


    print("CKA Linear:", np.mean(CKA_linear_list[:-1]))
    print("CKA Kernel:", np.mean(CKA_kernel_list[:-1]))



    # preds = np.array(preds)
    # trues = np.array(trues)
    # inputx = np.array(inputx)
    # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    inputx = np.concatenate(inputx, axis=0)  # if there is not that line, ignore this
    print('test shape:', preds.shape, trues.shape)

    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}'.format(mae, mse, rmse, smape))


    ##计算每个样本的MAE和MSE指标
    samplemae = SampleMAE(preds, trues)
    samplemse = SampleMSE(preds, trues)
    sample_results = np.array([samplemae, samplemse])
    np.save('{}_results_ETTm1_seasonality_{}_{}.npy'.format(args.model, args.seq_len, args.pred_len), sample_results)
    np.save('{}_input_ETTm1_seasonality_{}_{}.npy'.format(args.model, args.seq_len, args.pred_len), inputx)
    np.save('{}_preds_ETTm1_seasonality_{}_{}.npy'.format(args.model, args.seq_len, args.pred_len), preds)
    np.save('{}_trues_ETTm1_seasonality_{}_{}.npy'.format(args.model, args.seq_len, args.pred_len), trues)



    return mse, mae


def vali_GPT4TS(model, vali_data, vali_loader, criterion, args, device):
    total_loss = []
    patchtst_loss = []
    ##这里为什么只对in_layer和out_layer进行eval
    # model.in_layer.eval()
    # model.out_layer.eval()
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            output_dic = model(batch_x, test=False)
            outputs = output_dic["outputs"]
            #patchtst_outputs = output_dic["outputs"]
            #res_outputs = output_dic["res_outputs"]

            ###为了变长的输出
            outputs = outputs[:, :args.pred_len, :]
            #patchtst_outputs = patchtst_outputs[:, :args.pred_len, :]


            batch_y = batch_y[:, -args.pred_len:, :].to(device)


            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            #patchtst_pred = outputs.detach().cpu()

            loss = criterion(pred, true)
            #loss = 0
            #time_loss = criterion(patchtst_pred, true)
            # loss = criterion(pred, trend_truth) + criterion(res_pred, res_truth)
            # time_loss = criterion(patchtst_pred, trend_truth) + criterion(res_pred, res_truth)
            total_loss.append(loss)
            #patchtst_loss.append(time_loss)
    #patchtst_loss = np.average(patchtst_loss)
    total_loss = np.average(total_loss)
    model.train()
    return total_loss

# def test(model, test_data, test_loader, args, device, itr):
#     preds = []
#     trues = []
#     folder_path = './../visualization/ETTm1336/'
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#
#     model.eval()
#     with torch.no_grad():
#         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
#
#             batch_x = batch_x.float().to(device)
#             batch_y = batch_y.float()
#
#             output_dic = model(batch_x, itr)
#             outputs = output_dic["outputs"]
#             trend_outputs = output_dic["trend_outputs"]
#             sea_outputs = output_dic["sea_outputs"]
#             res_outputs = output_dic["res_outputs"]
#             trend = output_dic["trend"]
#             sea = output_dic["sea"]
#             res = output_dic["res"]
#             scale_input = output_dic["scale_x"]
#
#
#             # encoder - decoder
#             outputs = outputs[:, -args.pred_len:, :]
#             batch_y = batch_y[:, -args.pred_len:, :].to(device)
#
#             pred = outputs.detach().cpu().numpy()
#             true = batch_y.detach().cpu().numpy()
#
#             if i % 5 == 0:
#                 input = batch_x.detach().cpu().numpy()
#                 trend = trend.detach().cpu().numpy()
#                 sea = sea.detach().cpu().numpy()
#                 res = res.detach().cpu().numpy()
#                 scale_input = scale_input.detach().cpu().numpy()
#                 trend_outputs = trend_outputs.detach().cpu().numpy()
#                 sea_outputs = sea_outputs.detach().cpu().numpy()
#                 res_outputs = res_outputs.detach().cpu().numpy()
#
#                 gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
#                 pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
#                 visual_decomp(input[0,:,-1], trend[0,:,-1], sea[0,:,-1], res[0,:,-1], os.path.join(folder_path, str(i) + 'decomposition.pdf'))
#                 visual_decomp(true[0,:,-1], trend_outputs[0,:,-1], sea_outputs[0,:,-1], res_outputs[0,:,-1], os.path.join(folder_path, str(i) + 'predict_decomposition.pdf'))
#                 visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
#
#
#             preds.append(pred)
#             trues.append(true)
#
#     preds = np.array(preds)
#     trues = np.array(trues)
#     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
#     trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
#     print('test shape:', preds.shape, trues.shape)
#
#     mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
#     print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}'.format(mae, mse, rmse, smape))
#
#     return mse, mae







def MASE(x, freq, pred, true):
    masep = np.mean(np.abs(x[:, freq:] - x[:, :-freq]))
    return np.mean(np.abs(pred - true) / (masep + 1e-8))


def test_GPT4TS(model, test_data, test_loader, args, device):
    preds = []
    trues = []
    inputx = []
    mae_list = []
    mse_list = []
    CKA_linear_list = []
    CKA_kernel_list = []
    gpt_last_hidden = []
    folder_path = './../visualization/ETTm1_720/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            output_dic = model(batch_x, test=True)
            outputs = output_dic["outputs"]
            #patchtst_outputs = output_dic["patchtst_outputs"]

            cka_kernel_metric = output_dic["cka_kernel_metric"]
            cka_linear_metric = output_dic["cka_linear_metric"]
            CKA_linear_list.append(cka_linear_metric)
            CKA_kernel_list.append(cka_kernel_metric)
            gpt_last_hidden.append(output_dic["gpt_last_hidden"].detach().cpu().numpy())

            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            input = batch_x.detach().cpu().numpy()

            preds.append(pred)
            trues.append(true)
            inputx.append(input)


    print("CKA Linear:", np.mean(CKA_linear_list[:-1]))
    print("CKA Kernel:", np.mean(CKA_kernel_list[:-1]))



    # preds = np.array(preds)
    # trues = np.array(trues)
    # inputx = np.array(inputx)
    # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    inputx = np.concatenate(inputx, axis=0)  # if there is not that line, ignore this
    last_hidden = np.concatenate(gpt_last_hidden, axis=0)
    print('test shape:', preds.shape, trues.shape)

    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}'.format(mae, mse, rmse, smape))

    np.save('{}_input_ETTm1_{}_{}.npy'.format(args.model, args.seq_len, args.pred_len), inputx)
    np.save('{}_gpt_hidden_ETTm1_{}_{}.npy'.format(args.model, args.seq_len, args.pred_len), last_hidden)



    # ##计算每个样本的MAE和MSE指标
    # samplemae = SampleMAE(preds, trues)
    # samplemse = SampleMSE(preds, trues)
    # sample_results = np.array([samplemae, samplemse])
    # np.save('{}_results_ETTm1_seasonality_{}_{}.npy'.format(args.model, args.seq_len, args.pred_len), sample_results)
    # np.save('{}_input_ETTm1_seasonality_{}_{}.npy'.format(args.model, args.seq_len, args.pred_len), inputx)
    # np.save('{}_preds_ETTm1_seasonality_{}_{}.npy'.format(args.model, args.seq_len, args.pred_len), preds)
    # np.save('{}_trues_ETTm1_seasonality_{}_{}.npy'.format(args.model, args.seq_len, args.pred_len), trues)
    #
    return mse, mae

