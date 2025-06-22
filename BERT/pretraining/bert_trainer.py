import torch
from tqdm.auto import tqdm

def train(model, train_loader, criterion, optimizer, scheduler, config):
    model.train()
    total_loss, total_mlm_loss, total_nsp_loss = 0, 0, 0

    for batch in tqdm(train_loader, desc='Training...'):
        input_ids, input_mask, segment_ids, next_sentence_labels, \
        mlm_pred_positions, mlm_pred_labels, mlm_weights = [item.to(config.device) for item in batch]

        optimizer.zero_grad()

        mlm_logits, nsp_logits = model(input_ids, segment_ids, input_mask)

        mlm_pred_labels[mlm_weights == 0] = -100
        mlm_target_labels = torch.full(input_ids.shape, -100, dtype=torch.long, device=config.device)
        mlm_target_labels.scatter_(1, mlm_pred_positions, mlm_pred_labels)

        mlm_loss = criterion(mlm_logits.view(-1, config.vocab_size), mlm_target_labels.view(-1)) # mlm loss
        nsp_loss = criterion(nsp_logits, next_sentence_labels) # nsp loss
        loss = mlm_loss + nsp_loss # total loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip) # clip = 0.1

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_mlm_loss += mlm_loss.item()
        total_nsp_loss += nsp_loss.item()
    return total_loss / len(train_loader), total_mlm_loss / len(train_loader), total_nsp_loss / len(train_loader)
