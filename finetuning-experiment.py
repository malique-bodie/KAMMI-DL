
#@title Huggingface Pretrained Wrapper

# Updated for cs grid

"""
This is script is a simple HuggingFace wrapper around a HyenaDNA model, to enable a one click example
of how to load the pretrained weights and get embeddings.

It will instantiate a HyenaDNA model (model class is in the `standalone_hyenadna.py`), and handle the downloading of pretrained weights from HuggingFace.

Check out the colab notebook for a simpler and more complete walk through of how to use HyenaDNA with pretrained weights.

"""


import json
import os
import subprocess
import torch
# import transformers
from transformers import PreTrainedModel, AutoModelForSequenceClassification, AutoTokenizer
import re
from standalone_hyenadna import HyenaDNAModel
from standalone_hyenadna import CharacterTokenizer
import numpy as np


# helper 1
def inject_substring(orig_str):
    """Hack to handle matching keys between models trained with and without
    gradient checkpointing."""

    # modify for mixer keys
    pattern = r"\.mixer"
    injection = ".mixer.layer"

    modified_string = re.sub(pattern, injection, orig_str)

    # modify for mlp keys
    pattern = r"\.mlp"
    injection = ".mlp.layer"

    modified_string = re.sub(pattern, injection, modified_string)

    return modified_string

# helper 2
def load_weights(scratch_dict, pretrained_dict, checkpointing=False):
    """Loads pretrained (backbone only) weights into the scratch state dict."""

    # loop thru state dict of scratch
    # find the corresponding weights in the loaded model, and set it

    # need to do some state dict "surgery"
    for key, value in scratch_dict.items():
        if 'backbone' in key:
            # the state dicts differ by one prefix, '.model', so we add that
            key_loaded = 'model.' + key
            # breakpoint()
            # need to add an extra ".layer" in key
            if checkpointing:
                key_loaded = inject_substring(key_loaded)
            try:
                scratch_dict[key] = pretrained_dict[key_loaded]
            except:
                raise Exception('key mismatch in the state dicts!')

    # scratch_dict has been updated
    return scratch_dict

class HyenaDNAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    base_model_prefix = "hyenadna"

    def __init__(self, config):
        pass

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    @classmethod
    def from_pretrained(cls,
                        path,
                        model_name,
                        download=False,
                        config=None,
                        device='cpu',
                        use_head=False,
                        n_classes=2,
                      ):
        # first check if it is a local path
        pretrained_model_name_or_path = os.path.join(path, model_name)
        if os.path.isdir(pretrained_model_name_or_path) and download == False:
            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))
        else:
            hf_url = f'https://huggingface.co/LongSafari/{model_name}'

            subprocess.run(f'rm -rf {pretrained_model_name_or_path}', shell=True)
            command = f'mkdir -p {path} && cd {path} && git lfs install && git clone {hf_url}'
            subprocess.run(command, shell=True)

            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))

        scratch_model = HyenaDNAModel(**config, use_head=use_head, n_classes=n_classes)  # the new model format
        loaded_ckpt = torch.load(
            os.path.join(pretrained_model_name_or_path, 'weights.ckpt'),
            map_location=torch.device(device)
        )

        # need to load weights slightly different if using gradient checkpointing
        if config.get("checkpoint_mixer", False):
            checkpointing = config["checkpoint_mixer"] == True or config["checkpoint_mixer"] == True
        else:
            checkpointing = False

        # grab state dict from both and load weights
        state_dict = load_weights(scratch_model.state_dict(), loaded_ckpt['state_dict'], checkpointing=checkpointing)

        # scratch model has now been updated
        scratch_model.load_state_dict(state_dict)
        print("Loaded pretrained weights ok!")
        return scratch_model




####################################################################################################




"""# Inference (450k to 1M tokens)!

If all you're interested in is getting embeddings on long DNA sequences
(inference), then we can do that right here in Colab!


*   We provide an example how to load the weights from Huggingface.
*   On the free tier, which uses a
T4 GPU w/16GB of memory, we can process 450k tokens / nucleotides.
*   For processing 1M tokens, you'll need an A100, which Colab offers as a paid tier.
*   (Don't forget to run the entire notebook above too)

--

To pretrain or fine-tune the 1M long sequence model (8 layers, d_model=256),
you'll need 8 A100s 80GB, and all that code is in the main repo!
"""

#@title Single example
import json
import os
import subprocess
# import transformers
from transformers import PreTrainedModel
from transformers import TrainingArguments, Trainer, logging

import os
os.environ["WANDB_MODE"]="offline"

# Finetuning using HF/Ilana's notebook
def finetune_single():

    print("testing finetune_single()")

    '''
    this selects which backbone to use, and grabs weights/ config from HF
    4 options:
      'hyenadna-tiny-1k-seqlen'   # fine-tune on colab ok
      'hyenadna-small-32k-seqlen'
      'hyenadna-medium-160k-seqlen'  # inference only on colab
      'hyenadna-medium-450k-seqlen'  # inference only on colab
      'hyenadna-large-1m-seqlen'  # inference only on colab
    '''

    # hardcoded for now
    # checkpoint = 'LongSafari/hyenadna-tiny-1k-seqlen-hf'
    # max_length = 1024

    checkpoint = 'LongSafari/hyenadna-large-1m-seqlen-hf'
    max_length = 1_000_000

    print('finetune_single', checkpoint)

    # max_lengths = {
    #     'hyenadna-tiny-1k-seqlen': 1024,
    #     'hyenadna-small-32k-seqlen': 32768,
    #     'hyenadna-medium-160k-seqlen': 160000,
    #     'hyenadna-medium-450k-seqlen': 450000,  # T4 up to here
    #     'hyenadna-large-1m-seqlen': 1_000_000,  # only A100 (paid tier)
    # }

    # bfloat16 for better speed and reduced memory usage
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    # Needed to support higher training bz
    model.config.pad_token_id = tokenizer.pad_token_id
    
    import pandas as pd
    from datasets import Dataset

    train_df = pd.read_csv("/ifs/CS/replicated/home/inguyen4/Desktop/cs2952g/hyena-dna/data/GUE/tf/0/train.csv", header=0)
    train_sequence = train_df['sequence']
    train_sequence = train_sequence.tolist()
    train_tokenized = tokenizer(train_sequence)["input_ids"]
    train_labels = train_df['label']
    train_labels = train_labels.tolist()


    test_df = pd.read_csv("/ifs/CS/replicated/home/inguyen4/Desktop/cs2952g/hyena-dna/data/GUE/tf/0/test.csv", header=0)
    test_sequence = test_df['sequence']
    test_sequence = test_sequence.tolist()
    test_tokenized = tokenizer(test_sequence)["input_ids"]
    test_labels = test_df['label']
    test_labels = test_labels.tolist()

    # Create a dataset for training
    ds_train = Dataset.from_dict({"input_ids": train_tokenized, "labels": train_labels})
    ds_test = Dataset.from_dict({"input_ids": test_tokenized, "labels": test_labels})
    ds_train.set_format("pt")
    ds_test.set_format("pt")

    args = {
    "output_dir": "tmp",
    "evaluation_strategy": "epoch",
    "save_strategy": "no",
    "num_train_epochs": 100,
    "per_device_train_batch_size": 256,
    "learning_rate": 2e-5,
    "per_device_eval_batch_size": 1,
    }

    training_args = TrainingArguments(**args)

    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score

    # Define the metric for the evaluation using the accuracy score
    def compute_metrics(eval_pred):
        """Computes accuracy score for binary classification"""
        predictions = np.argmax(eval_pred.predictions, axis=-1)
        references = eval_pred.label_ids
        r={'accuracy_score': accuracy_score(references, predictions), 
        'f1_score': f1_score(references, predictions, average="macro", zero_division=0),
        'matthews_corrcoef': matthews_corrcoef(references, predictions),
        'precision_score': precision_score(references, predictions, average="macro", zero_division=0),
        'recall_score': recall_score(references, predictions, average="macro", zero_division=0),
        }
        return r

    

    trainer = Trainer(model=model,
                    args=training_args,
                    train_dataset=ds_train,
                    eval_dataset=ds_test,
                    compute_metrics=compute_metrics,)
    result = trainer.train()

    print(result)

# uncomment to run! (to get embeddings)
finetune_single()

# def inference_single():

#     '''
#     this selects which backbone to use, and grabs weights/ config from HF
#     4 options:
#       'hyenadna-tiny-1k-seqlen'   # fine-tune on colab ok
#       'hyenadna-small-32k-seqlen'
#       'hyenadna-medium-160k-seqlen'  # inference only on colab
#       'hyenadna-medium-450k-seqlen'  # inference only on colab
#       'hyenadna-large-1m-seqlen'  # inference only on colab
#     '''

#     # you only need to select which model to use here, we'll do the rest!
#     pretrained_model_name = 'hyenadna-small-32k-seqlen'

#     max_lengths = {
#         'hyenadna-tiny-1k-seqlen': 1024,
#         'hyenadna-small-32k-seqlen': 32768,
#         'hyenadna-medium-160k-seqlen': 160000,
#         'hyenadna-medium-450k-seqlen': 450000,  # T4 up to here
#         'hyenadna-large-1m-seqlen': 1_000_000,  # only A100 (paid tier)
#     }

#     max_length = max_lengths[pretrained_model_name]  # auto selects

#     # data settings:
#     use_padding = True
#     rc_aug = False  # reverse complement augmentation
#     add_eos = False  # add end of sentence token

#     # we need these for the decoder head, if using
#     use_head = False
#     n_classes = 2  # not used for embeddings only

#     # you can override with your own backbone config here if you want,
#     # otherwise we'll load the HF one in None
#     backbone_cfg = None

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print("Using device:", device)

#     # instantiate the model (pretrained here)
#     if pretrained_model_name in ['hyenadna-tiny-1k-seqlen',
#                                  'hyenadna-small-32k-seqlen',
#                                  'hyenadna-medium-160k-seqlen',
#                                  'hyenadna-medium-450k-seqlen',
#                                  'hyenadna-large-1m-seqlen']:
#         # use the pretrained Huggingface wrapper instead
#         model = HyenaDNAPreTrainedModel.from_pretrained(
#             './checkpoints',
#             pretrained_model_name,
#             download=True,
#             config=backbone_cfg,
#             device=device,
#             use_head=use_head,
#             n_classes=n_classes,
#         )

#     # from scratch
#     elif pretrained_model_name is None:
#         model = HyenaDNAModel(**backbone_cfg, use_head=use_head, n_classes=n_classes)

#     # create tokenizer
#     tokenizer = CharacterTokenizer(
#         characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
#         model_max_length=max_length + 2,  # to account for special tokens, like EOS
#         add_special_tokens=False,  # we handle special tokens elsewhere
#         padding_side='left', # since HyenaDNA is causal, we pad on the left
#     )

#     #### Single embedding example ####

#     # create a sample 450k long, prepare
#     sequence = 'ACTG' * int(max_length/4)
#     tok_seq = tokenizer(sequence)
#     tok_seq = tok_seq["input_ids"]  # grab ids

#     # place on device, convert to tensor
#     tok_seq = torch.LongTensor(tok_seq).unsqueeze(0)  # unsqueeze for batch dim
#     tok_seq = tok_seq.to(device)

#     # prep model and forward
#     model.to(device)
#     model.eval()
#     with torch.inference_mode():
#         embeddings = model(tok_seq)

#     print(embeddings.shape)  # embeddings here!

# # uncomment to run! (to get embeddings)
# inference_single()


# to run this, just call:
    # python huggingface.py
