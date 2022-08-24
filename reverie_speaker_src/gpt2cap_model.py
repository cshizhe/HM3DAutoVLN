import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

from transformers import GPT2Tokenizer, GPT2LMHeadModel

class ReveriePanoObject2DGPT2CapModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.device = 'cuda:%d'%config.GPUID if torch.cuda.is_available() else 'cpu'

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.STOP_TOKEN_ID = self.tokenizer.eos_token_id
        self.vocab_size = len(self.tokenizer.get_vocab())

        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=self.tokenizer.eos_token_id)
        self.freeze_gpt()

        self.obj_type_embedding = nn.Embedding(2, self.config.HIDDEN_SIZE)
        self.obj_img_linear = nn.Sequential(
            nn.Linear(self.config.OBJIMG_FT_SIZE, self.config.HIDDEN_SIZE),
            nn.LayerNorm(self.config.HIDDEN_SIZE),
        )
        self.obj_name_linear = nn.Sequential(
            nn.Linear(self.config.OBJNAME_FT_SIZE, self.config.HIDDEN_SIZE),
            nn.LayerNorm(self.config.HIDDEN_SIZE),
        )
        self.obj_loc_linear = nn.Sequential(
            nn.Linear(4, self.config.HIDDEN_SIZE),
            nn.LayerNorm(self.config.HIDDEN_SIZE),
        )
        self.obj_size_linear = nn.Sequential(
            nn.Linear(3, self.config.HIDDEN_SIZE),
            nn.LayerNorm(self.config.HIDDEN_SIZE),
        )
        
        if self.config.USE_VIEW_FT:
            self.view_img_linear = nn.Sequential(
                nn.Linear(self.config.VIEWIMG_FT_SIZE, self.config.HIDDEN_SIZE),
                nn.LayerNorm(self.config.HIDDEN_SIZE),
            )
            self.view_loc_linear = nn.Sequential(
                nn.Linear(4, self.config.HIDDEN_SIZE),
                nn.LayerNorm(self.config.HIDDEN_SIZE),
            )
            self.view_type_embedding = nn.Embedding(2, self.config.HIDDEN_SIZE)

        # caption decoder
        if self.config.ENC_LAYERS > 0:
            enc_layer = nn.TransformerEncoderLayer(
                self.config.HIDDEN_SIZE, 8, batch_first=True,
            )
            self.prefix_map = nn.TransformerEncoder(
                enc_layer,
                self.config.ENC_LAYERS
            )
        else:
            self.prefix_map = None

    def freeze_gpt(self):
        if self.config.GPT_FREEZE:
            for name, param in self.gpt.named_parameters():
                param.requires_grad = False

    def prepare_batch_data(self, batch):
        new_batch = {}
        for k, v in batch.items():
            if k in ['names', 'ref_caps']:
                new_batch[k] = v
            else:
                new_batch[k] = v.to(self.device)
        return new_batch

    def forward_encoder(self, batch):
        batch_size = len(batch['obj_fts'])

        a = self.config.OBJIMG_FT_SIZE
        b = a + self.config.OBJNAME_FT_SIZE
        obj_embeds = self.obj_img_linear(batch['obj_fts'][..., :a]) + \
                     self.obj_name_linear(batch['obj_fts'][..., a: b]) + \
                     self.obj_loc_linear(batch['obj_fts'][..., b: b+4]) + \
                     self.obj_size_linear(batch['obj_fts'][..., b+4: ]) + \
                     self.obj_type_embedding(batch['obj_types'])
        
        if self.config.USE_VIEW_FT:
            view_embeds = self.view_img_linear(batch['view_fts'][..., :-4]) + \
                        self.view_loc_linear(batch['view_fts'][..., -4:]) + \
                        self.view_type_embedding(batch['view_types'])
            input_embeds = torch.cat([view_embeds, obj_embeds], dim=1)
            input_masks = torch.zeros(batch_size, 36, dtype=torch.bool).to(self.device)
            input_masks = torch.cat([input_masks, batch['obj_masks']], dim=1)
        else:
            input_embeds = obj_embeds
            input_masks = batch['obj_masks']

        if self.prefix_map is not None:
            out_embeds = self.prefix_map(
                input_embeds, src_key_padding_mask=input_masks,
            )
        else:
            out_embeds = input_embeds

        obj_masks0 = batch['obj_masks'].logical_not()
        if self.config.USE_VIEW_FT:
            if self.config.USE_CTX_OBJS:
                ctx_obj_embeds = torch.sum(out_embeds[:, 36:] * obj_masks0.unsqueeze(2), 1) \
                                 / obj_masks0.sum(1, keepdims=True)
                # tgt view embeds, ctx obj embeds, tgt obj embeds
                prefix_embeds = torch.stack(
                    [torch.mean(out_embeds[:, :36], 1), out_embeds[:, 0], 
                    ctx_obj_embeds, out_embeds[:, 36]], 1
                )
            else:
                prefix_embeds = torch.stack(
                    [torch.mean(out_embeds[:, :36], 1), out_embeds[:, 0], 
                    out_embeds[:, 36]], 1
                )
        else:
            if self.config.USE_CTX_OBJS:
                ctx_obj_embeds = torch.sum(out_embeds * obj_masks0.unsqueeze(2), 1) \
                                 / obj_masks0.sum(1, keepdims=True)
                # tgt view embeds, ctx obj embeds, tgt obj embeds
                prefix_embeds = torch.stack(
                    [ctx_obj_embeds, out_embeds[:, 0]], 1
                )
            else:
                prefix_embeds = out_embeds[:, 0]

        return prefix_embeds

    def forward(self, batch, compute_loss=False):
        batch = self.prepare_batch_data(batch)

        enc_embeds = self.forward_encoder(batch)
        max_enc_len = enc_embeds.size(1)
    
        cap_ids = batch['cap_ids']
        cap_lens = batch['cap_lens']
        batch_size, max_cap_len = cap_ids.size()

        cap_embeds = self.gpt.transformer.wte(cap_ids[:, :-1])
        input_embeds = torch.cat([enc_embeds, cap_embeds], 1)

        cap_logits = self.gpt(inputs_embeds=input_embeds).logits
        cap_logits = cap_logits[:, max_enc_len-1:]
        
        if compute_loss:
            loss = self.compute_cap_loss(cap_logits, batch['cap_ids'], batch['cap_lens'])
            return cap_logits, loss

        return cap_logits

    def compute_cap_loss(self, cap_logits, cap_ids, cap_lens):
        batch_size, max_cap_len, _ = cap_logits.size()
        cap_losses = F.cross_entropy(
            cap_logits.permute(0, 2, 1), # (N, C, L)
            cap_ids,
            reduction='none'
        )   # (N, L)
        cap_masks = torch.arange(max_cap_len).long().repeat(batch_size, 1).to(self.device) < cap_lens.unsqueeze(1)
        cap_masks = cap_masks.float()
        cap_loss = torch.sum(cap_losses * cap_masks) / torch.sum(cap_masks)
        return cap_loss

    def greedy_inference(self, batch, max_txt_len=100):
        # TODO: optimize speed (save cache)
        batch = self.prepare_batch_data(batch)

        enc_embeds = self.forward_encoder(batch)
        batch_size, max_enc_len, _ = enc_embeds.size()

        pred_cap_ids = None
        unfinished = torch.ones(batch_size).bool().to(self.device)
        generated = enc_embeds
        for t in range(max_txt_len):
            outputs = self.gpt(inputs_embeds=generated)
            logits = outputs.logits[:, -1]

            next_tokens = torch.argmax(logits, dim=1).unsqueeze(1)
            next_token_embeds = self.gpt.transformer.wte(next_tokens)
            if pred_cap_ids is None:
                pred_cap_ids = next_tokens
            else:
                pred_cap_ids = torch.cat([pred_cap_ids, next_tokens], dim=1)
            generated = torch.cat([generated, next_token_embeds], dim=1)
            
            unfinished = unfinished & (next_tokens != self.STOP_TOKEN_ID)
            if torch.sum(unfinished).item() == 0:
                break

        pred_cap_ids = pred_cap_ids.data.cpu().numpy()
        pred_caps = []
        for sids in pred_cap_ids:
            cut_sids = []
            for sid in sids:
                if sid == self.STOP_TOKEN_ID:
                    break
                cut_sids.append(sid)
            pred_caps.append(self.tokenizer.decode(cut_sids, skip_special_tokens=True))
        return pred_caps

    def beam_inference(self, batch, beam_size=5, max_txt_len=100):
        batch = self.prepare_batch_data(batch)

        enc_embeds = self.forward_encoder(batch)
        batch_size, max_enc_len, _ = enc_embeds.size()

        pred_caps = []
        # TODO: one sample per time for simplicity, need speedup
        for i in range(batch_size):
            tokens = None
            scores = None
            seq_lengths = torch.ones(beam_size, device=self.device)
            is_stopped = torch.zeros(beam_size, device=self.device, dtype=torch.bool)

            generated = enc_embeds[i].unsqueeze(0)

            for t in range(max_txt_len):
                outputs = self.gpt(inputs_embeds=generated)
                logits = outputs.logits[:, -1]
                logprobs = logits.softmax(-1).log()
                if scores is None:
                    scores, next_tokens = logprobs.topk(beam_size, -1)
                    generated = generated.expand(beam_size, *generated.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    tokens = next_tokens
                else:
                    logprobs[is_stopped] = -float('inf')
                    logprobs[is_stopped, 0] = 0
                    scores_sum = scores[:, None] + logprobs
                    seq_lengths[~is_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                        beam_size, -1
                    )
                    next_tokens_source = next_tokens // scores_sum.size(1)  # size(1)=vocab_size
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.size(1)
                    next_tokens = next_tokens.unsqueeze(1)
                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                    generated = generated[next_tokens_source]
                    scores = scores_sum_average * seq_lengths
                    is_stopped = is_stopped[next_tokens_source]

                next_token_embed = self.gpt.transformer.wte(next_tokens)
                generated = torch.cat([generated, next_token_embed], dim=1)
                is_stopped = is_stopped + next_tokens.eq(self.STOP_TOKEN_ID).squeeze()
                if is_stopped.all():
                    break

            scores = scores / seq_lengths
            output_list = tokens.cpu().numpy()
            output_texts = [
                self.tokenizer.decode(output[: int(length)], skip_special_tokens=True)
                for output, length in zip(output_list, seq_lengths)
            ]
            order = scores.argsort(descending=True)
            output_texts = [output_texts[i] for i in order]
            output_scores = scores[order].data.cpu().numpy().tolist()
            pred_caps.append(
                [(score, text.strip()) for score, text in zip(output_scores, output_texts)]
            )
        return pred_caps

    def save(self, ckpt_file):
        # do not save clip parameters if we freeze clip
        state_dict = self.state_dict()
        new_state_dict = {}
        if self.config.GPT_FREEZE:
            for name, param in state_dict.items():
                if not name.startswith('gpt'):
                    new_state_dict[name] = param
            state_dict = new_state_dict

        ckpt_dir = os.path.dirname(ckpt_file)
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(state_dict, ckpt_file)

    def load(self, ckpt_file):
        state_dict = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        print('load %d params' % (len(state_dict)))
        self.load_state_dict(state_dict, strict=True)


class LSTMDecoder(nn.Module):
    def __init__(self, num_words, hidden_size) -> None:
        super().__init__()
        self.num_words = num_words
        self.hidden_size = hidden_size

        self.bos = nn.Parameter(torch.zeros(hidden_size).float())
        self.word_embedding = nn.Embedding(num_words, hidden_size)
        self.lstm_cell = nn.LSTMCell(hidden_size*2, hidden_size)
        self.attn_key_fc = nn.Linear(hidden_size, hidden_size)
        self.attn_query_fc = nn.Linear(hidden_size, hidden_size)
        self.attn_w = nn.Linear(hidden_size, 1)

    def forward(self, enc_embeds, word_ids):
        batch_size = enc_embeds.size(0)

        attn_keys = self.attn_key_fc(enc_embeds)
        word_embeds = self.word_embedding(word_ids)
        word_embeds = torch.cat([einops.repeat(self.bos, 'd -> b 1 d', b=batch_size), word_embeds], 1)
        
        h = torch.zeros(batch_size, self.hidden_size).to(enc_embeds.device)
        c = torch.zeros(batch_size, self.hidden_size).to(enc_embeds.device)

        cap_logits = []
        for t in range(word_embeds.size(1)):
            attn_queries = einops.repeat(self.attn_query_fc(h), 'b d -> b k d', k=enc_embeds.size(1))
            attn_scores = F.softmax(self.attn_w(torch.tanh(attn_queries + attn_keys)).squeeze(2), dim=1)
            attn_embeds = torch.einsum('bk,bkd->bd', attn_scores, enc_embeds)
            input_embeds = torch.cat([word_embeds[:, t], attn_embeds], 1)

            h, c = self.lstm_cell(input_embeds, (h, c))
            cap_logits.append(F.linear(h, self.word_embedding.weight))
        cap_logits = torch.stack(cap_logits, 1)
        return cap_logits   # (b, t, nw)
        

class ReveriePanoObject2DLSTMCapModel(ReveriePanoObject2DGPT2CapModel):
    def __init__(self, config):
        super().__init__(config)
        del self.gpt
        self.decoder = LSTMDecoder(self.vocab_size, self.config.HIDDEN_SIZE)

    def forward(self, batch, compute_loss=False):
        batch = self.prepare_batch_data(batch)

        enc_embeds = self.forward_encoder(batch)
        max_enc_len = enc_embeds.size(1)
    
        cap_ids = batch['cap_ids']  # w1, ..., wn, <eos> (no <bos>)
        cap_lens = batch['cap_lens']
        batch_size, max_cap_len = cap_ids.size()

        cap_logits = self.decoder(enc_embeds, cap_ids[:, :-1])
        
        if compute_loss:
            loss = self.compute_cap_loss(cap_logits, batch['cap_ids'], batch['cap_lens'])
            return cap_logits, loss

        return cap_logits

    def greedy_inference(self, batch, max_txt_len=100):
        # TODO: optimize speed (save cache)
        batch = self.prepare_batch_data(batch)

        enc_embeds = self.forward_encoder(batch)
        batch_size, max_enc_len, _ = enc_embeds.size()

        attn_keys = self.decoder.attn_key_fc(enc_embeds)
        h = torch.zeros(batch_size, self.decoder.hidden_size).to(enc_embeds.device)
        c = torch.zeros(batch_size, self.decoder.hidden_size).to(enc_embeds.device)

        pred_cap_ids = None
        unfinished = torch.ones(batch_size).bool().to(self.device)
        word_embeds = einops.repeat(self.decoder.bos, 'd -> b d', b=batch_size)

        for t in range(max_txt_len):
            attn_queries = einops.repeat(self.decoder.attn_query_fc(h), 'b d -> b k d', k=max_enc_len)
            attn_scores = F.softmax(self.decoder.attn_w(torch.tanh(attn_queries + attn_keys)).squeeze(2), dim=1)
            attn_embeds = torch.einsum('bk,bkd->bd', attn_scores, enc_embeds)
            input_embeds = torch.cat([word_embeds, attn_embeds], 1)

            h, c = self.decoder.lstm_cell(input_embeds, (h, c))
            logits = F.linear(h, self.decoder.word_embedding.weight)

            next_tokens = torch.argmax(logits, dim=1)
            word_embeds = self.decoder.word_embedding(next_tokens)
            next_tokens = next_tokens.unsqueeze(1)
            if pred_cap_ids is None:
                pred_cap_ids = next_tokens
            else:
                pred_cap_ids = torch.cat([pred_cap_ids, next_tokens], dim=1)
            
            unfinished = unfinished & (next_tokens != self.STOP_TOKEN_ID)
            if torch.sum(unfinished).item() == 0:
                break

        pred_cap_ids = pred_cap_ids.data.cpu().numpy()
        pred_caps = []
        for sids in pred_cap_ids:
            cut_sids = []
            for sid in sids:
                if sid == self.STOP_TOKEN_ID:
                    break
                cut_sids.append(sid)
            pred_caps.append(self.tokenizer.decode(cut_sids, skip_special_tokens=True))
        return pred_caps

