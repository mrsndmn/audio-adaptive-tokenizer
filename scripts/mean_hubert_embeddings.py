import os
import shutil
from tqdm.auto import tqdm

import torch

if __name__ == "__main__":
    source_dir = "data/audio_segments_embeddings/"

    target_dir = "data/audio_segments_embeddings_mean/"
    if os.path.exists(target_dir) and os.path.isdir(target_dir):
        shutil.rmtree(target_dir)

    os.makedirs(target_dir, exist_ok=True)

    for embeding_file_name in tqdm(os.listdir(source_dir)):
        full_embedding_path = os.path.join(source_dir, embeding_file_name)
        embedings_list = torch.load(full_embedding_path, map_location='cpu', weights_only=True)
        mean_embeddings = [ x.mean(dim=1, keepdim=True).to(torch.float32) for x in embedings_list ]
        averaged_hubert_embeddings_t = torch.cat(mean_embeddings, dim=1) # [ 1, seq_length, 768 ]

        target_file = os.path.join(target_dir, embeding_file_name)
        torch.save(averaged_hubert_embeddings_t, target_file)
