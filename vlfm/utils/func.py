import torch

def longclip_pos_embeddings(model, new_max_token):
    text_model = model.text_model
    # Extract positional embeddings
    pos_embeddings_pre = text_model.embeddings.position_embedding.weight
    length, dim = pos_embeddings_pre.shape
    keep_len = 20
    new_length = 4*length - 3*keep_len
    if new_length < new_max_token:
        raise ValueError("new_max_token is too large")
    pos_embeddings_new = torch.zeros([new_max_token, dim], dtype=pos_embeddings_pre.dtype)
    for i in range(keep_len):
        pos_embeddings_new[i] = pos_embeddings_pre[i]
    for i in range(length-1-keep_len):
        pos_embeddings_new[4*i + keep_len] = pos_embeddings_pre[i + keep_len]
        pos_embeddings_new[4*i + 1 + keep_len] = 3*pos_embeddings_pre[i + keep_len]/4 + 1*pos_embeddings_pre[i+1+keep_len]/4
        pos_embeddings_new[4*i + 2+keep_len] = 2*pos_embeddings_pre[i+keep_len]/4 + 2*pos_embeddings_pre[i+1+keep_len]/4
        pos_embeddings_new[4*i + 3+keep_len] = 1*pos_embeddings_pre[i+keep_len]/4 + 3*pos_embeddings_pre[i+1+keep_len]/4
    pos_embeddings_new[4*length -3*keep_len - 4] = pos_embeddings_pre[length-1] + 0*(pos_embeddings_pre[length-1] - pos_embeddings_pre[length-2])/4
    pos_embeddings_new[4*length -3*keep_len - 3] = pos_embeddings_pre[length-1] + 1*(pos_embeddings_pre[length-1] - pos_embeddings_pre[length-2])/4
    pos_embeddings_new[4*length -3*keep_len - 2] = pos_embeddings_pre[length-1] + 2*(pos_embeddings_pre[length-1] - pos_embeddings_pre[length-2])/4
    pos_embeddings_new[4*length -3*keep_len - 1] = pos_embeddings_pre[length-1] + 3*(pos_embeddings_pre[length-1] - pos_embeddings_pre[length-2])/4
    text_model.embeddings.position_embedding.weight = torch.nn.Parameter(pos_embeddings_new)
    # Set position_ids if the model uses them
    if hasattr(text_model.embeddings, 'position_ids'):
        text_model.embeddings.position_ids = torch.arange(0, new_max_token).unsqueeze(0)
    else:
        text_model.register_buffer('position_ids', torch.arange(0, new_max_token).unsqueeze(0))

