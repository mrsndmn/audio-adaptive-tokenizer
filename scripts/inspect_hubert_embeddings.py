import os
import torch

if __name__ == "__main__":

    embeddings_base_path = "data/audio_segments_embeddings_mean_tokenized/"
    mean_tokenized_embeddings = os.listdir(embeddings_base_path)
    for hubert_audio_embedding in mean_tokenized_embeddings[:10]:
        audio_embeds = torch.load(os.path.join(embeddings_base_path, hubert_audio_embedding), weights_only=True) # [ 1, tokens_count (seq_len), 768 ]
        print(f"mean: {audio_embeds.mean(dim=-1).mean().item():.2f} norm: {audio_embeds.norm(2, dim=-1).mean().item():.2f}")
        breakpoint()