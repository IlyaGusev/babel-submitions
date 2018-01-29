import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, n_layers=3, dropout=0.1):
        super(EncoderRNN, self).__init__()
        
        num_directions = 2
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
       
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        
    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        packed = pack(embedded, input_lengths)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, output_lengths = unpack(outputs)
        n = hidden[0].size(0)
        hidden = (torch.cat([hidden[0][0:n:2], hidden[0][1:n:2]], 2), torch.cat([hidden[1][0:n:2], hidden[1][1:n:2]], 2))
        return outputs, hidden
     
class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.sm = nn.Softmax(dim=1)
        
        self.out = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, decoder_rnn_output, encoder_outputs):
        max_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        
        decoder_rnn_output = decoder_rnn_output.transpose(0, 1)
        energy = self.attn(encoder_outputs).view(batch_size, self.hidden_size, max_len)
        attn_energies = decoder_rnn_output.bmm(energy).transpose(0, 1).squeeze(0)  # S = B x L
        
        attn_weights = self.sm(attn_energies).unsqueeze(1) # S = B x 1 x L
        encoder_context = attn_weights.bmm(encoder_outputs.transpose(0, 1)).transpose(0, 1) # S = 1 x B x N
        
        concat_context = torch.cat([encoder_context, decoder_rnn_output.transpose(0, 1)], 2)
        context = self.tanh(self.out(concat_context))

        return context, attn_weights.squeeze(1)
    
class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size, n_layers=3, dropout=0.1, max_length=50):
        super(AttnDecoderRNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_length = max_length
        
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.attn = Attn(hidden_size)
        self.rnn = nn.LSTM(hidden_size + embedding_dim, hidden_size, n_layers, dropout=dropout)

    def step(self, input_seq, hidden, encoder_outputs, context):
        # hidden: S = n_layers x B x N
        # encoder_outputs: S = L x B x N 
        embedded = self.embedding(input_seq).unsqueeze(0) # S = 1 x B x E
    
        # Combine embedded input word and attended context, run through RNN (input feeding)
        rnn_input = torch.cat((embedded, context), 2)
        output, hidden = self.rnn(rnn_input, hidden)
        
         # Calculate attention weights and apply to encoder outputs
        output, attn_weights = self.attn(output, encoder_outputs) 
        # output: # S = 1 x B x N
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights
    
    def init_state(self, batch_size):
        initial_input = Variable(torch.LongTensor([1 for _ in range(batch_size)]), requires_grad=False)
        initial_input = initial_input.cuda() if use_cuda else initial_input
        
        initial_context = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False).unsqueeze(0)
        initial_context = initial_context.cuda() if use_cuda else initial_context
        
        return initial_input, initial_context
    
    def forward(self, inputs, input_lengths, hidden, encoder_outputs, initial_input, initial_context):
        batch_size = encoder_outputs.size(1)
        max_input_length = max(input_lengths)
        max_encoder_length = encoder_outputs.size(0)
        
        outputs = Variable(torch.zeros(max_input_length + 1, batch_size, self.hidden_size), requires_grad=False)
        outputs = outputs.cuda() if use_cuda else outputs
        
        attn_weights = Variable(torch.zeros(max_input_length + 1, batch_size, max_encoder_length), requires_grad=False)
        attn_weights = attn_weights.cuda() if use_cuda else attn_weights
        
        context = initial_context
        for t in range(max_input_length + 1):
            if t != 0:
                current_input = inputs[t-1]
            else:
                current_input = initial_input
            context, hidden, attn = self.step(current_input, hidden, encoder_outputs, context)
            outputs[t] = context
            attn_weights[t] = attn
        
        return outputs, hidden, attn_weights
    
class Generator(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Generator, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.out = nn.Linear(hidden_size, output_size)
        self.sm = nn.LogSoftmax(dim=1)
    
    def forward(self, inputs):
        return self.sm(self.out(inputs))