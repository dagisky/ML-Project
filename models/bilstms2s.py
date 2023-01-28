import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module): # Encodes the question
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, hidden = self.rnn(embedded)
        hidden = self.fc(torch.cat((hidden[:,-2,:], hidden[:,-1,:]), dim = 1))
        
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + (dec_hid_dim * 2), dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)

    def forward(self, hidden, encoder_outputs):

        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(0).flatten(1, 2)
        hidden = hidden.unsqueeze(1).repeat(batch_size, src_len, 1)

        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = self.attn(torch.cat((hidden, encoder_outputs), dim = 2)) # (1, 86, 1024)

        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        #attention= [batch size, src len]

        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.attention = attention

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, dec_hid_dim, bidirectional = True)

        self.fc_out = nn.Linear(enc_hid_dim * 4, input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, encoder_hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)

        #embedded = [1, batch size, emb dim]

        a = self.attention(encoder_hidden, outputs)

        a = a.unsqueeze(1)
        weighted = torch.bmm(a, outputs)


        weighted = torch.cat((weighted, encoder_outputs[:,-1,:].unsqueeze(1)), dim=-1)
        prediction = self.fc_out(weighted.squeeze())
        return prediction

class Seq2Seq(nn.Module):
    def __init__(self, ctx_encoder, query_decoder, device):
        super().__init__()
        
        self.ctx_encoder = ctx_encoder
        self.query_decoder = query_decoder
        self.device = device
        
    def forward(self, src, query, teacher_forcing_ratio = 0.5):
        encoder_outputs, hidden = self.ctx_encoder(query)
        predictions = self.query_decoder(src, hidden, encoder_outputs)
        return F.softmax(predictions, dim = -1)



def create_model(vocab_size, device):
    INPUT_DIM = vocab_size
    ENC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    attention = Attention(ENC_HID_DIM, DEC_HID_DIM)
    dec = Decoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attention)

    model = Seq2Seq(enc, dec, device).to(device)
    return model