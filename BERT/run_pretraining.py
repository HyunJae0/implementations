import json
import random
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm.auto import tqdm
from get_pretraining_data import *
from bert_pretraining import *


def init_weights(module, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=initializer_range)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    config = PreTrainingBertConfig()
    tokenizer = HuggingFaceTokenizer(config.tokenizer_path)

    print('Loading dataset...')
    tokenizer = HuggingFaceTokenizer(config.tokenizer_path)
    wiki = load_dataset("lucadiliello/english_wikipedia", split='train')
    wiki = wiki.remove_columns(['filename', 'source_domain', 'title', 'url'])

    print('Preprocessing documents...')
    processed_wiki = wiki.map(process_batch, batched=True,
                              batch_size=1000, num_proc=4,
                              remove_columns=wiki.column_names)

    print('Tokenizing documents...')
    tokenization_wiki = processed_wiki.map(tokenize_batch, batched=True,
                                           batch_size=1000, num_proc=4,
                                           remove_columns=processed_wiki.column_names)

    print('Creating augmented instances (MLM & NSP)...')
    all_mlm_token_sequences, all_mlm_pred_positions, all_mlm_pred_labels, \
        all_segment_ids, all_nsp_labels = create_augmented_instances(
        tokenizer, tokenization_wiki, config)

    print(f'Padding and saving instances to {config.output_filename}...')
    add_padding_and_save_instances(
        all_mlm_token_sequences, all_mlm_pred_positions,
        all_mlm_pred_labels, all_segment_ids, all_nsp_labels,
        tokenizer, config)

    with open(config.output_filename, 'r') as f:
        instances = json.load(f)

    split_idx = int(len(instances['input_ids']) * 0.8)
    train_instances = {k: v[:split_idx] for k, v in instances.items()}
    valid_instances = {k: v[split_idx:] for k, v in instances.items()}

    train_loader = get_loader(train_instances, config, shuffle=True)
    valid_loader = get_loader(valid_instances, config, shuffle=False)

    print('Data Preparation Complete')

    print('Training Phase...')
    model = PreTrainingBertModel(config).to(config.device)
    model.apply(init_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    criterion = nn.CrossEntropyLoss().to(config.device)
    total_steps = len(train_loader) * config.epochs
    optimizer, scheduler = optimizer_and_scheduler(model, total_steps, config)

    best_loss = float('inf')
    for epoch in range(config.epochs):
        train_loss, train_mlm_loss, train_nsp_loss = train(model, train_loader, criterion, optimizer, scheduler, config)
        valid_loss, valid_mlm_loss, valid_nsp_loss = evaluate(model, valid_loader, criterion, config)

        print(f'Train Loss: {train_loss:.4f} | MLM Loss: {train_mlm_loss:.4f} | NSP Loss: {train_nsp_loss:.4f}')
        print(f'Valid Loss: {valid_loss:.4f} | MLM Loss: {valid_mlm_loss:.4f} | NSP Loss: {valid_nsp_loss:.4f}')

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'best_model-{0}.pt'.format(valid_loss))
    print('Training Complete...')
