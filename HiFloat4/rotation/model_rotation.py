import logging
from rotation import rotation_utils
from rotation import hadamard_utils
import math
import tqdm
import fast_hadamard_transform


# 分离embeddings和lm_head的函数
def separate_embeddings_and_lm_head(model):
    import torch.nn as nn
    model.config.tie_word_embeddings = False
    if hasattr(model, 'lm_head') and hasattr(model, 'model'):
        if hasattr(model.model, 'embed_tokens'):
            # 获取嵌入层的权重
            embed_weight = model.model.embed_tokens.weight.data.clone()

            # 确保 lm_head 有独立的权重
            if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
                # 权重共享，需要断开
                model.lm_head.weight = nn.Parameter(embed_weight.clone())
                logging.info("Successfully disconnected tied word embeddings")
            else:
                logging.info("Word embeddings are already disconnected")
        else:
            logging.warning("Could not find embed_tokens layer")
    else:
        logging.warning("Could not find lm_head or model layers")


def prepare_model(model, rot_block_size=0):
    if model.config.tie_word_embeddings:  # 断开权重共享 针对 Llama-3.2
        logging.info("Tying word embeddings is not supported for rotation, disabling it.")
        separate_embeddings_and_lm_head(model)

    rotation_utils.fuse_layer_norms(model)
    rotation_utils.rotate_model(model, rot_block_size=rot_block_size)
    rotation_utils.cleanup_memory(verbos=True)

    return model
