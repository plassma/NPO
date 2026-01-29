from functools import partial

import jax
import jax.numpy as jnp

from .BaseEnergy import BaseEnergyClass


class CountdownEnergyClass(BaseEnergyClass):

    def __init__(self, config):
        super().__init__(config)
        

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy(self, meta_graph, bins, node_gr_idx, A=1.0, B=1.2):
        tokens = jnp.atleast_1d(jnp.asarray(bins)).reshape(-1).astype(jnp.int32)
        node_gr_idx = jnp.atleast_1d(jnp.asarray(node_gr_idx)).reshape(-1).astype(jnp.int32)

        numbers = jnp.atleast_2d(jnp.asarray(meta_graph.globals["numbers"][:-1]))
        targets = jnp.ravel(jnp.asarray(meta_graph.globals["target"][:-1]))

        n_graph = numbers.shape[0]
        n_operands = numbers.shape[1]
        targets = jnp.broadcast_to(targets, (n_graph,))

        op_add = n_operands
        op_sub = n_operands + 1
        op_mul = n_operands + 2
        op_div = n_operands + 3
        eos_tok = n_operands + 4

        stack0 = jnp.zeros((n_graph, n_operands), dtype=numbers.dtype)
        depth0 = jnp.zeros((n_graph,), dtype=jnp.int32)
        eos0 = jnp.zeros((n_graph,), dtype=jnp.bool_)

        def step(carry, inp):
            stack, depth, eos_seen = carry
            tok, g = inp

            def update(_):
                depth_g = depth[g]
                is_operand = tok < n_operands
                is_op = (tok == op_add) | (tok == op_sub) | (tok == op_mul) | (tok == op_div)
                is_eos = tok == eos_tok

                def do_operand(_):
                    value = numbers[g, tok]

                    def push(_):
                        stack2 = stack.at[g, depth_g].set(value)
                        depth2 = depth.at[g].set(depth_g + 1)
                        return stack2, depth2, eos_seen

                    return jax.lax.cond(depth_g < n_operands, push, lambda _: (stack, depth, eos_seen), operand=None)

                def do_op(_):
                    def apply_op(_):
                        a = stack[g, depth_g - 2]
                        b = stack[g, depth_g - 1]

                        def add(_):
                            return a + b

                        def sub(_):
                            return a - b

                        def mul(_):
                            return a * b

                        def div(_):
                            return jnp.where(b != 0, a // b, 0)

                        res = jax.lax.switch(tok - op_add, [add, sub, mul, div], operand=None)
                        stack2 = stack.at[g, depth_g - 2].set(res)
                        depth2 = depth.at[g].set(depth_g - 1)
                        return stack2, depth2, eos_seen

                    return jax.lax.cond(depth_g >= 2, apply_op, lambda _: (stack, depth, eos_seen), operand=None)

                def do_eos(_):
                    eos2 = eos_seen.at[g].set(True)
                    return stack, depth, eos2

                def do_nothing(_):
                    return stack, depth, eos_seen

                return jax.lax.cond(
                    is_operand,
                    do_operand,
                    lambda _: jax.lax.cond(
                        is_op,
                        do_op,
                        lambda _: jax.lax.cond(is_eos, do_eos, do_nothing, operand=None),
                        operand=None,
                    ),
                    operand=None,
                )

            new_carry = jax.lax.cond(eos_seen[g], lambda _: (stack, depth, eos_seen), update, operand=None)
            return new_carry, None

        (stack, depth, _), _ = jax.lax.scan(step, (stack0, depth0, eos0), (tokens, node_gr_idx))

        top_idx = jnp.maximum(depth - 1, 0)
        expr_values = stack[jnp.arange(n_graph), top_idx]
        expr_values = jnp.where(depth > 0, expr_values, 0)

        expr_values = expr_values.astype(jnp.float32)
        targets = targets.astype(jnp.float32)
        energy = jnp.abs(expr_values - targets)

        n_node = jnp.asarray(meta_graph.n_node).reshape(-1)
        energy = jnp.where(n_node > 0, energy, 0.0)

        total_energy = jnp.array([energy[0], 0])[:, None].astype(jnp.float32)
        breakdown = {"total_energy": total_energy}        
        return total_energy, breakdown, jnp.zeros(bins.shape[0])

    def calculate_relaxed_Energy(self, H_graph, bins, node_gr_idx, A=1.0, B=1.2):
        return self.calculate_Energy(H_graph, bins, node_gr_idx, A=A, B=B)

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_loss(self, H_graph, logits, node_gr_idx, A=1.0, B=1.2):
        return self.calculate_Energy(H_graph, logits, node_gr_idx, A=A, B=B)
