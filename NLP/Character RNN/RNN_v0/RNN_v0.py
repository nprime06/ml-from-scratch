import sys
sys.path.append('../../../Modules')
import numpy as np
import math
import pandas as pd
from tqdm import tqdm
from Sigmoid import softmax

# dont mind that this version isnt very clean

dtype = np.float32

# architecture hyperparams
hidden_size = 256

# training hyperparams
epochs = 200
batch_size = 32
retain_steps = 4
seq_len = 70
learning_rate = dtype(0.0002)
beta1 = dtype(0.9)
beta2 = dtype(0.999)
eps = dtype(1e-8)


def forward_training(w_xh, w_hh, w_ho, b_h, b_o, oh_input, init_hidden_state): # oh_input (transposed): seq_len, batch_size, vocab_size
	hidden_state = init_hidden_state # (batch_size, hidden_size)
	hidden_states = np.zeros((seq_len + 1, batch_size, hidden_size), dtype = dtype)
	hidden_states[0] += init_hidden_state
	oh_output = np.zeros_like(oh_input)

	for i in range(seq_len):
		hidden_states[i + 1] += np.tanh(oh_input[i] @ w_xh.T + hidden_states[i] @ w_hh.T + b_h)
		oh_output[i] += (hidden_states[i] @ w_ho.T + b_o)

	return hidden_states, oh_output # seq + 1, N, hidden; seq, N, vocab


def forward_generation(w_xh, w_hh, w_ho, b_h, b_o, characters, temperature, oh_prompt = None): # prompt: prompt_length, 1, vocab
	hidden_state = np.zeros((1, hidden_size), dtype = dtype)

	if oh_prompt is not None:
		for char in oh_prompt:
			hidden_state = np.tanh(char @ w_xh.T + hidden_state @ w_hh.T + b_h)

	outputs = []
	for _ in range(characters):
		out_logits = hidden_state @ w_ho.T + b_o
		out_probabilities = softmax(out_logits / temperature, dtype = np.float64).ravel() # float for np random choice
		vocab_size = len(out_probabilities)
		next_char_idx = np.random.choice(vocab_size, p = out_probabilities)
		outputs.append(next_char_idx)
		next_char = np.eye(vocab_size, dtype = np.int32)[next_char_idx].reshape(1, -1)
		hidden_state = np.tanh(next_char @ w_xh.T + hidden_state @ w_hh.T + b_h)

	return np.array(outputs) # flat characters, (not encoded)


def gradient_calculation(w_xh, w_hh, w_ho, b_h, b_o, x, h, z, max_norm = 3.0): # train on last i - 1 in sequence
	grad_z = np.zeros_like(z) # seq, N, vocab
	grad_h = np.zeros_like(h) # seq + 1, N, hidden
	grad_phi = np.zeros_like(h) 

	grad_z[1:] += (softmax(z[:-1], axis = 2) - x[1:])
	for i in reversed(range(seq_len)):
		grad_phi[i + 1] += grad_h[i + 1] * (1 - (h[i + 1] * h[i + 1])) # fills [1:]
		grad_h[i] += (grad_phi[i + 1] @ w_hh + grad_z[i] @ w_ho) # fills [:-1]
		# grad phi i+1 <- h i+1
		# h i <- phi i+1

	N_group = (seq_len - 1) * batch_size
	vocab_size = z.shape[-1]

	grad_w_ho = grad_z[1:].reshape(-1, vocab_size).T @ h[1:-1].reshape(-1, hidden_size) / N_group # sum H[1:-1] and grad_z[1:]
	# grad_z: seq - 1, N, vocab
	# h:      seq - 1, N, hidden
	# w_ho: vocab, hidden

	grad_w_hh = grad_phi[1:-1].reshape(-1, hidden_size).T @ h[:-2].reshape(-1, hidden_size) / N_group # sum grad phi[1:-1], h[:-2]
	# grad phi: seq - 1, N, hidden
	# h: seq - 1, N, hidden
	# w hh: hidden, hidden

	grad_w_xh = grad_phi[1:-1].reshape(-1, hidden_size).T @ x[:-1].reshape(-1, vocab_size) / N_group # sum grad phi[1:-1], x[:-1]
	# grad phi: seq - 1, N, hidden
	# x: seq - 1, N, vocab
	# w xh: hidden, vocab

	grad_b_o = np.mean(grad_z[1:], axis = (0, 1), dtype = dtype).reshape(1, -1) # sum grad z [1:]
	grad_b_h = np.mean(grad_phi[1:-1], axis = (0, 1), dtype = dtype).reshape(1, -1) # sum grad phi [1:-1]
	# questionable shape

	# grad clip
	grads = [grad_w_xh, grad_w_hh, grad_w_ho, grad_b_h, grad_b_o]
	max_norm = dtype(max_norm)
	total_norm_sq = 0
	for grad in grads:
		total_norm_sq += np.sum(grad * grad, dtype = dtype)
	total_norm = np.sqrt(total_norm_sq)
	if total_norm > max_norm: 
		scale = max_norm / total_norm
		for grad in grads:
			grad *= scale

	return grad_w_xh, grad_w_hh, grad_w_ho, grad_b_h, grad_b_o # normed by (seq - 1) * N


def main():
	with open('random.txt', 'r', encoding = 'utf-8') as f:
		text = f.read()
	
	text_size = len(text) # ~5 million
	vocab = sorted(set(text))
	vocab_size = len(vocab)
	char2idx = {char: idx for idx, char in enumerate(vocab)}
	idx2char = {idx: char for idx, char in enumerate(vocab)}

	encoded_text = np.array([char2idx[c] for c in text], dtype = np.int32) # flat
	# oh_text = np.eye(len(vocab), dtype = np.int32)[encoded_text]

	# init weights (He, just guessing), biases
	#w_xh = (np.sqrt(2. / (vocab_size)) * np.random.randn(hidden_size, vocab_size)).astype(dtype = dtype)
	#w_hh = (np.sqrt(2. / (hidden_size)) * np.random.randn(hidden_size, hidden_size)).astype(dtype = dtype)
	#w_ho = (np.sqrt(2. / (hidden_size)) * np.random.randn(vocab_size, hidden_size)).astype(dtype = dtype)
	w_xh = (0.01 * np.random.randn(hidden_size, vocab_size)).astype(dtype = dtype)
	w_hh = (0.01 * np.random.randn(hidden_size, hidden_size)).astype(dtype = dtype)
	w_ho = (0.01 * np.random.randn(vocab_size, hidden_size)).astype(dtype = dtype)
	b_h = np.zeros((1, hidden_size), dtype = dtype)
	b_o = np.zeros((1, vocab_size), dtype = dtype)
	params = [w_xh, w_hh, w_ho, b_h, b_o]
	moment1 = [np.zeros_like(param) for param in params]
	moment2 = [np.zeros_like(param) for param in params]
	t = 0

	# prepare data to put into training
	# clip to multiple of : batch_size * retain_steps * seq_len
	seq_count = text_size // (batch_size * retain_steps * seq_len)
	batched_text = encoded_text[:(seq_count * batch_size * retain_steps * seq_len)].reshape(-1, batch_size, retain_steps, seq_len).transpose(0, 2, 1, 3)
	# am i retarded

	for epoch in tqdm(range(epochs)):
		# balls and cock!

		for step in batched_text:
			init_hidden_state = np.zeros((batch_size, hidden_size), dtype = dtype)
			for batch in step: # batch: (N, seq_len)
				oh_batch = np.eye(vocab_size, dtype = dtype)[batch.T] # seq, N, vocab
				hidden_states, oh_output = forward_training(w_xh, w_hh, w_ho, b_h, b_o, oh_batch, init_hidden_state) # send batch forward seminiferously!
				init_hidden_state = hidden_states[-1]
				# grad_w_xh, grad_w_hh, grad_w_ho, grad_b_h, grad_b_o = gradient_calculation(w_xh, w_hh, w_ho, b_h, b_o, oh_batch, hidden_states, oh_output)
				grads = list(gradient_calculation(w_xh, w_hh, w_ho, b_h, b_o, oh_batch, hidden_states, oh_output))

				#print(grad_w_xh.dtype, grad_w_hh.dtype, grad_w_ho.dtype, grad_b_h.dtype, grad_b_o.dtype)

				# SGD
				'''
				w_xh -= learning_rate * grad_w_xh
				w_hh -= learning_rate * grad_w_hh
				w_ho -= learning_rate * grad_w_ho
				b_h -= learning_rate * grad_b_h
				b_o -= learning_rate * grad_b_o
				'''

				# adam
				t += 1

				for i, (param, grad) in enumerate(zip(params, grads)):
					moment1[i] = beta1 * moment1[i] + (1 - beta1) * grad
					moment2[i] = beta2 * moment2[i] + (1 - beta2) * np.power(grad, 2)

					alpha_t = learning_rate * dtype(math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t))) # see Kingma et al.
					param -= alpha_t * moment1[i] / (np.sqrt(moment2[i]) + eps)


		for i in range(5):
			sample = forward_generation(w_xh, w_hh, w_ho, b_h, b_o, 100, 0.8)
			translated_sample = [idx2char[idx] for idx in sample]
			print(f'epoch {epoch} sample {i}', ''.join(translated_sample))


# def forward_training(w_xh, w_hh, w_ho, b_h, b_o, oh_input, init_hidden_state): -> hidden_states, oh_output
# def forward_generation(w_xh, w_hh, w_ho, b_h, b_o, characters, temperature, oh_prompt = None): -> np.array(outputs) indices
# def gradient_calculation(w_xh, w_hh, w_ho, b_h, b_o, x, h, z, max_norm = 5.0): -> grad_w_xh, grad_w_hh, grad_w_ho, grad_b_h, grad_b_o


if __name__ == '__main__': 
	main()