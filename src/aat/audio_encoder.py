# import torch.nn as nn
# import torch


# class AudioEncoder(nn.Module):
#     def __init__(self, melspec_features, hidden_dim=512):

#         self.melspec_features = melspec_features
#         self.hidden_dim = hidden_dim

#         return

#     def forward(self, waveform, segments_boarders_padded, segments_boarders_attention_mask):
#         """Compute waveform embeddings by segments boarders from segments_boarders_padded

#         Args:
#             waveform (bs, n_frames): batch of raw waveforms
#             segments_boarders_padded (n_boarders): Segments lengths
#             segments_boarders_attention_mask (n_boarders): Segments lengths padding mask
#         """

#         return

