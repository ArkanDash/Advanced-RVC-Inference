import torch

import numpy as np
import torch.nn.functional as F

from librosa.util import pad_center
from scipy.signal import get_window

class STFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=512, win_length=None, window="hann", pad_mode="reflect"):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.pad_amount = int(self.filter_length / 2)
        self.cutoff = int(self.filter_length / 2 + 1)
        self.win_length = win_length
        self.pad_mode = pad_mode
        self.hann_window = {}

        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        fourier_basis = np.vstack([np.real(fourier_basis[:self.cutoff, :]), np.imag(fourier_basis[:self.cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis)
        inverse_basis = torch.FloatTensor(np.linalg.pinv(fourier_basis))

        if win_length is None or not win_length: win_length = filter_length
        assert filter_length >= win_length

        fft_window = torch.from_numpy(pad_center(get_window(window, win_length, fftbins=True), size=filter_length)).float()
        forward_basis *= fft_window
        inverse_basis = (inverse_basis.T * fft_window).T

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())
        self.register_buffer("fft_window", fft_window.float())

    def transform(self, input_data, eps=1e-9, return_phase=False, center=True):
        if center: input_data = F.pad(input_data, (self.pad_amount, self.pad_amount), mode=self.pad_mode)
        forward_transform = (self.forward_basis @ input_data.unfold(1, self.filter_length, self.hop_length).permute(0, 2, 1))

        real_part = forward_transform[:, :self.cutoff, :]
        imag_part = forward_transform[:, self.cutoff:, :]
        magnitude = (real_part**2 + imag_part**2 + eps).sqrt()

        if return_phase:
            phase = imag_part.data.atan2(real_part.data)
            return magnitude, phase

        return magnitude

    def inverse(self, magnitude, phase):
        cat = torch.cat([magnitude * phase.cos(), magnitude * phase.sin()], dim=1)
        fold = torch.nn.Fold(output_size=(1, (cat.size(-1) - 1) * self.hop_length + self.filter_length), kernel_size=(1, self.filter_length), stride=(1, self.hop_length))

        inverse_transform = fold(self.inverse_basis @ cat)[:, 0, 0, self.pad_amount : -self.pad_amount]
        window_square_sum = fold(self.fft_window.cpu().pow(2).repeat(cat.size(-1), 1).T.unsqueeze(0))[:, 0, 0, self.pad_amount : -self.pad_amount].to(cat.device) if str(cat.device).startswith("ocl") else fold(self.fft_window.pow(2).repeat(cat.size(-1), 1).T.unsqueeze(0))[:, 0, 0, self.pad_amount : -self.pad_amount]

        return inverse_transform / window_square_sum

class GRU(torch.nn.RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0.0, bidirectional=False, device=None, dtype=None):
        super().__init__("GRU", input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, device=device, dtype=dtype)

    @staticmethod
    def _gru_cell(x, hx, weight_ih, bias_ih, weight_hh, bias_hh):
        gate_x = F.linear(x, weight_ih, bias_ih)
        gate_h = F.linear(hx, weight_hh, bias_hh)

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = (i_r + h_r).sigmoid()
        inputgate = (i_i + h_i).sigmoid()
        newgate = (i_n + resetgate * h_n).tanh()

        hy = newgate + inputgate * (hx - newgate)
        return hy

    def _gru_layer(self, x, hx, weights):
        weight_ih, weight_hh, bias_ih, bias_hh = weights
        outputs = []

        for x_t in x.unbind(1):
            hx = self._gru_cell(x_t, hx, weight_ih, bias_ih, weight_hh, bias_hh)
            outputs.append(hx)

        return torch.stack(outputs, dim=1), hx

    def _gru(self, x, hx):
        if not self.batch_first: x = x.permute(1, 0, 2)
        num_directions = 2 if self.bidirectional else 1

        h_n = []
        output_fwd, output_bwd = x, x

        for layer in range(self.num_layers):
            fwd_idx = layer * num_directions
            bwd_idx = fwd_idx + 1 if self.bidirectional else None

            weights_fwd = self._get_weights(fwd_idx)
            h_fwd = hx[fwd_idx]

            out_fwd, h_out_fwd = self._gru_layer(output_fwd, h_fwd, weights_fwd)
            h_n.append(h_out_fwd)

            if self.bidirectional:
                weights_bwd = self._get_weights(bwd_idx)
                h_bwd = hx[bwd_idx]

                reversed_input = torch.flip(output_bwd, dims=[1])
                out_bwd, h_out_bwd = self._gru_layer(reversed_input, h_bwd, weights_bwd)

                out_bwd = torch.flip(out_bwd, dims=[1])
                h_n.append(h_out_bwd)

                output_fwd = torch.cat([out_fwd, out_bwd], dim=2)
                output_bwd = output_fwd
            else: output_fwd = out_fwd

            if layer < self.num_layers - 1 and self.dropout > 0:
                output_fwd = F.dropout(output_fwd, p=self.dropout, training=self.training)
                if self.bidirectional: output_bwd = output_fwd

        output = output_fwd
        h_n = torch.stack(h_n, dim=0)

        if not self.batch_first: output = output.permute(1, 0, 2)
        return output, h_n

    def _get_weights(self, layer_idx):
        weights = self._all_weights[layer_idx]

        weight_ih = getattr(self, weights[0])
        weight_hh = getattr(self, weights[1])

        bias_ih = getattr(self, weights[2]) if self.bias else None
        bias_hh = getattr(self, weights[3]) if self.bias else None

        return weight_ih, weight_hh, bias_ih, bias_hh

    def forward(self, input, hx=None):
        if input.dim() != 3: raise ValueError

        batch_size = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1

        if hx is None: hx = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, dtype=input.dtype, device=input.device)

        self.check_forward_args(input, hx, batch_sizes=None)
        return self._gru(input, hx)