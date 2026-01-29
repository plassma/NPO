
import jax
from flax import struct
from jax import lax
from jax import numpy as jnp

from Problems.GraphWithMeta import GraphWithMeta

SEP_TOKEN = 0
@struct.dataclass
class CountdownGraph(GraphWithMeta):
	
	def embed_nodes(self, diff_model, X_t, energy_per_node, t_idx_per_node):
		dtype = jnp.bfloat16 if diff_model.bfloat16 else jnp.float32
		
		prefix_int = jnp.concatenate([self.globals["target"][:-1], self.globals["numbers"][:-1]]) # todo: probably needs pos embedding, could add separator token here, do not use num-encoder for it
		prefix_int_feats = jnp.array([prefix_int, prefix_int ** 2, jnp.log(prefix_int + 1), jnp.sqrt(prefix_int + 1)]).T # todo: add modulo feats here?
		prefix_int_emb = diff_model.number_encoder(prefix_int_feats.astype(dtype))
		symbols = diff_model.symbol_encoder(jnp.concat([jnp.array([SEP_TOKEN]), X_t[..., 0]]))
		seq = jnp.concat([prefix_int_emb, symbols], axis=0)
		
		return seq
	
	def get_vocab_size(self):
		return make_token_ids(self.meta["num_operands"])[-1]  # V token is vocab size

	def get_prior_logits(self, shape, soft=False):
		base_logits = jnp.repeat(jnp.log(1 / self.get_vocab_size()), self.get_vocab_size())
		target_shape = shape[:-1] + (base_logits.shape[-1],)
		return jnp.broadcast_to(base_logits[None, None], target_shape)

	def get_mask(self):
		mask = (jnp.arange(self.meta["cabinets"])[None, :] < self.globals["classes_per_node"][:, None])
		return mask * 1.0
	
	
	def masked_logits_from_scores(self, scores):
		without_prefix = scores[self.meta["num_operands"] + 2 :]  # remove prefix tokens
		if len(scores.shape) == 2:
			without_prefix = without_prefix[:, None, :]
		return jax.nn.log_softmax(without_prefix, axis=-1)
	
	def sample_from_logits(self, logits, key):
		# logits: [L, N, V]; scan_sample_from_logits expects [L, V]
		reduce_dim = False
		if len(logits.shape) == 2:
			logits = logits[:, None, :]
			reduce_dim = True
		n_samples = logits.shape[1]
		keys = jax.random.split(key, n_samples)

		def _sample_one(k, logits_LV):
			toks, _, _ = scan_sample_from_logits(
				k,
				logits_LV,
				self.globals["numbers"][:-1],
				self.meta["num_operands"],
				self.meta["num_operands"],
			)
			return toks
		
		toks_NL = jax.vmap(_sample_one, in_axes=(0, 1), out_axes=0)(keys, logits)
		toks = jnp.swapaxes(toks_NL, 0, 1)  # [L, N]
		X = toks[..., None]  # [L, N, 1]
		one_hot = jax.nn.one_hot(toks, num_classes=self.get_vocab_size())

		if reduce_dim:
			X = X[:, 0, :]
			one_hot = one_hot[:, 0, :]

		return X.astype(jnp.int32), one_hot # todo plassma: return actual logits!
	
	def compute_solution_prob_stats(self, spin_logits_next):
		print("Warning: using mocked solution prob stats in CountdownGraph.")
		return 0, 0
	
	def calc_mean_prob(self, spin_log_probs):
		print("Warning: check calc_mean_prob_again in CountdownGraph!")
		return spin_log_probs.mean()

def make_token_ids(Nmax):
	IDX0 = 0
	OP_ADD = Nmax + 0
	OP_SUB = Nmax + 1
	OP_MUL = Nmax + 2
	OP_DIV = Nmax + 3
	EOS    = Nmax + 4
	PAD    = Nmax + 5
	V = Nmax + 6
	return IDX0, OP_ADD, OP_SUB, OP_MUL, OP_DIV, EOS, PAD, V

def tokenize_rpn(Nmax, solution_rpn):
	"""
	Converts RPN tokens (string or list) into the corresponding token IDs used by the model.
	Operands are in [0, Nmax-1], operators are in {Nmax, Nmax+1, Nmax+2, Nmax+3}, EOS is Nmax+4.
	"""
	IDX0, OP_ADD, OP_SUB, OP_MUL, OP_DIV, EOS, PAD, V = make_token_ids(Nmax)

	token_ids = []
	tokens = solution_rpn.split() if isinstance(solution_rpn, str) else solution_rpn
	for token in tokens:
		if token == '+':
			token_ids.append(OP_ADD)
		elif token == '-':
			token_ids.append(OP_SUB)
		elif token == '*':
			token_ids.append(OP_MUL)
		elif token == '/':
			token_ids.append(OP_DIV)
		else:
			# Operand
			operand_idx = int(token)
			token_ids.append(operand_idx)
	token_ids.append(EOS)
	return jnp.array(token_ids, dtype=jnp.int32)

def _masked_logits(logits, mask, neg_inf=-1e9):
	# mask: bool[V] True=allowed
	neg_inf = jnp.array(neg_inf, dtype=logits.dtype)
	return jnp.where(mask, logits, neg_inf)

def scan_sample_from_logits(
	rng,
	logits_LV,          # [L, V] from the model (unmasked)
	operands,           # [Nmax] int
	n_operands,         # scalar int <= Nmax
	Nmax,
	use_semantic_div_mask=True,
	return_full_logprobs=False,  # if True returns [L,V] log_probs and masks
):
	"""
	Samples a valid RPN program (subset of operands, exact int division) with hard masks.
	Returns:
	  toks:        [L] int32
	  logp_actions:[L] float32/float64  (masked policy log-prob of chosen action)
	  entropy:     [L] float            (masked policy entropy; useful for RL)
	  (optional) log_probs_LV: [L,V]
	  (optional) masks_LV:     [L,V] bool
	"""
	IDX0, OP_ADD, OP_SUB, OP_MUL, OP_DIV, EOS, PAD, V = make_token_ids(Nmax)
	L = logits_LV.shape[0]
	assert logits_LV.shape[1] == V, (logits_LV.shape, V)

	depth0 = jnp.int32(0)
	used0  = jnp.zeros((Nmax,), dtype=jnp.bool_)
	eos0   = jnp.bool_(False)

	# semantic stack (only meaningful if use_semantic_div_mask=True)
	stack0 = jnp.zeros((Nmax,), dtype=operands.dtype)

	def step(carry, inp):
		rng, depth, used, eos_seen, stack = carry
		i, logits_i = inp

		rng, subrng = jax.random.split(rng)

		allow0 = jnp.zeros((V,), dtype=jnp.bool_)

		def after_eos(_):
			# Only PAD allowed
			allow = allow0.at[PAD].set(True)
			masked = _masked_logits(logits_i, allow)
			log_probs = jax.nn.log_softmax(masked)
			tok = jnp.int32(PAD)
			logp = log_probs[PAD]
			# entropy under masked policy
			probs = jnp.exp(log_probs)
			ent = -jnp.sum(probs * log_probs)
			return tok, logp, ent, depth, used, eos_seen, stack, allow, log_probs

		def before_eos(_):
			idxs = jnp.arange(Nmax, dtype=jnp.int32)
			valid_idx = (idxs < n_operands) & (~used)
			allow = allow0.at[0:Nmax].set(valid_idx)

			# operators allowed if depth >= 2
			allow_op = (depth >= 2)
			allow = allow.at[OP_ADD].set(allow_op)
			allow = allow.at[OP_SUB].set(allow_op)
			allow = allow.at[OP_MUL].set(allow_op)

			# division optionally requires exact divisibility for current stack top
			def div_ok_fn(_):
				a = stack[depth - 2]
				b = stack[depth - 1]
				return (b != 0) & (a % b == 0)

			div_ok = lax.cond(
				use_semantic_div_mask & (depth >= 2),
				div_ok_fn,
				lambda _: jnp.bool_(True),
				operand=None
			)
			allow = allow.at[OP_DIV].set(allow_op & div_ok)

			# EOS allowed iff depth == 1 and used at least one operand
			used_any = jnp.any(used)
			allow = allow.at[EOS].set((depth == 1) & used_any)

			# PAD disallowed before EOS
			allow = allow.at[PAD].set(False)

			masked = _masked_logits(logits_i, allow)
			log_probs = jax.nn.log_softmax(masked)
			tok = jnp.int32(jax.random.categorical(subrng, masked))  # samples from masked policy
			logp = log_probs[tok]
			probs = jnp.exp(log_probs)
			ent = -jnp.sum(probs * log_probs)

			is_operand = tok < Nmax
			is_op = (tok == OP_ADD) | (tok == OP_SUB) | (tok == OP_MUL) | (tok == OP_DIV)
			is_eos = (tok == EOS)

			def do_operand(_):
				k = tok
				used2 = used.at[k].set(True)
				stack2 = stack.at[depth].set(operands[k])
				return depth + 1, used2, eos_seen, stack2

			def do_op(_):
				a = stack[depth - 2]
				b = stack[depth - 1]
				res = lax.switch(tok - OP_ADD, [
					lambda: a + b,
					lambda: a - b,
					lambda: a * b,
					lambda: a // b,   # safe because div mask ensured exact (if enabled)
				])
				stack2 = stack.at[depth - 2].set(res)
				return depth - 1, used, eos_seen, stack2

			def do_eos(_):
				return depth, used, jnp.bool_(True), stack

			depth2, used2, eos2, stack2 = lax.cond(
				is_operand,
				do_operand,
				lambda _: lax.cond(is_op, do_op, do_eos, operand=None),
				operand=None
			)

			return tok, logp, ent, depth2, used2, eos2, stack2, allow, log_probs

		tok, logp, ent, depth2, used2, eos2, stack2, allow, log_probs = lax.cond(
			eos_seen,
			after_eos,
			before_eos,
			operand=None
		)

		carry2 = (rng, depth2, used2, eos2, stack2)
		# We return allow/log_probs optionally (small V so OK)
		return carry2, (tok, logp, ent, allow, log_probs)

	carry0 = (rng, depth0, used0, eos0, stack0)
	inp = (jnp.arange(L, dtype=jnp.int32), logits_LV)
	_, outs = lax.scan(step, carry0, inp)

	toks, logp_actions, entropy, masks_LV, log_probs_LV = outs

	if return_full_logprobs:
		return toks, logp_actions, entropy, log_probs_LV, masks_LV
	else:
		return toks, logp_actions, entropy