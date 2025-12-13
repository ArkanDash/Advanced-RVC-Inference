import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from src.utils import to_local_average_cents # , to_viterbi_cents
from src.loss import bce
from src.constants import SAMPLE_RATE
from mir_eval.melody import raw_pitch_accuracy, to_cent_voicing, raw_chroma_accuracy, overall_accuracy
from mir_eval.melody import voicing_recall, voicing_false_alarm


def evaluate(dataset, model, hop_length, device, pitch_th=0.03):
    metrics = defaultdict(list)
    for data in dataset:
        mel = data['mel'].to(device)
        n_frames = mel.shape[-1]
        # print('n_frames', n_frames)
        # print('mel shape before padding', mel.shape)
        mel = F.pad(
            mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode='reflect'
        ).unsqueeze(0)
        # print('mel shape after padding', mel.shape)
        output_chunks = []
        pad_frames = mel.shape[-1]
        for start in range(0, pad_frames, 32000):
            # print('chunk @', start)
            end = min(start + 32000, pad_frames)
            mel_chunk = mel[..., start:end]
            assert (
                mel_chunk.shape[-1] % 32 == 0
            ), "chunk_size must be divisible by 32"
            # print(' before padding', mel_chunk.shape)
            # mel_chunk = F.pad(mel_chunk, (320, 320), mode="reflect")
            # print(' after padding', mel_chunk.shape)
            out_chunk = model(mel_chunk)
            # print(' result chunk', out_chunk.shape)
            # out_chunk = out_chunk[:, 320:-320, :]
            # print(' trimmed chunk', out_chunk.shape)
            output_chunks.append(out_chunk)

        pitch_pred = torch.cat(output_chunks, dim=1).squeeze(0)

        pitch_label = data['pitch'].to(device)
        pitch_pred = pitch_pred[ : pitch_label.shape[0]]
        loss = bce(pitch_pred, pitch_label)
        metrics['loss'].append(loss.item())

        cents_pred = to_local_average_cents(pitch_pred.cpu().numpy(), None, pitch_th)
        # cents_pred = to_viterbi_cents(pitch_pred.cpu().numpy())
        # print()
        cents_label = to_local_average_cents(pitch_label.cpu().numpy(), None, pitch_th)
        # cents_label = to_viterbi_cents(pitch_label.cpu().numpy())
        # print()

        freq_pred = np.array([10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in cents_pred])
        freq = np.array([10 * (2 ** (cent / 1200)) if cent else 0 for cent in cents_label])

        time_slice = np.array([i*hop_length*1000/SAMPLE_RATE for i in range(len(cents_label))])
        ref_v, ref_c, est_v, est_c = to_cent_voicing(time_slice, freq, time_slice, freq_pred)

        rpa = raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)
        rca = raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)
        oa = overall_accuracy(ref_v, ref_c, est_v, est_c)
        vfa = voicing_false_alarm(ref_v, est_v)
        vr = voicing_recall(ref_v, est_v)
        metrics['RPA'].append(rpa)
        metrics['RCA'].append(rca)
        metrics['OA'].append(oa)
        metrics['VFA'].append(vfa)
        metrics['VR'].append(vr)
        # if rpa < 0.9:
        print(data['file'], ':\t', rpa, '\t', oa)

    return metrics
