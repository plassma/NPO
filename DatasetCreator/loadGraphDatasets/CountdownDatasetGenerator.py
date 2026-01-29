import random

import numpy as np
from jraph import GraphsTuple

from Problems.CountdownGraph import CountdownGraph, tokenize_rpn

from .BaseDatasetGenerator import BaseDatasetGenerator


def generate_solvable_instance(num_operands=5):
    """
    Generates a solvable Countdown instance by building an expression tree 
    bottom-up and evaluating it.
    """
    ops = ['+', '-', '*', '/']
    # 1. Start with random leaf nodes (numbers)
    nums = [random.randint(1, 10) for _ in range(num_operands)]
    current_values = list(nums)
    # Track RPN tokens using absolute operand indices (0..num_operands-1).
    current_rpn = [[str(i)] for i in range(num_operands)]
    
    # 2. Simulate operations to find a valid target
    history = []
    while len(current_values) > 1:
        # Pick two random numbers
        a_idx = random.randint(0, len(current_values)-1)
        a = current_values.pop(a_idx)
        a_rpn = current_rpn.pop(a_idx)
        b_idx = random.randint(0, len(current_values)-1)
        b = current_values.pop(b_idx)
        b_rpn = current_rpn.pop(b_idx)
        op = random.choice(ops)
        
        # Avoid division by zero or messy fractions for integer-only tasks
        if op == '/':
            if b == 0 or a % b != 0: 
                op = '+' # Fallback
                val = a + b
            else:
                val = a // b
        elif op == '-':
            val = a - b # Negative numbers are usually allowed in intermediate steps
        elif op == '*':
            val = a * b
        else:
            val = a + b
            
        current_values.append(val)
        rpn_expr = a_rpn + b_rpn + [op]
        current_rpn.append(rpn_expr)
        history.append(f"({a} {op} {b} = {val})")

    target = current_values[0]
    solution_rpn = " ".join(current_rpn[0])
    return {"inputs": nums, "target": target, "solution_trace": history, "solution_trace_rpn": solution_rpn}

def generate_solvable_instance_bounded_target(num_operands=5, target_min=-100, target_max=100):
    """
    Generates a solvable Countdown instance with a target within specified bounds.
    """
    while True:
        instance = generate_solvable_instance(num_operands)
        if target_min <= instance["target"] <= target_max:
            return instance


class CountdownDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, config):
        super().__init__(config)
        
        self.graph_config = {
			"n_train": 10000,
			"n_val": 1000,
			"n_test": 1000,
        }
        print(f'\nGenerating Countdown {self.mode} dataset "{self.dataset_name}" with {self.graph_config[f"n_{self.mode}"]} instances!\n')

    def _generate_dataset(self):
        num_operands = 4
        target_abs = 100
        op_abs = 10
        solutions = {
			"Energies": [],
			"H_graphs": [],
			"gs_bins": [],
			"graph_sizes": [],
		}

        for idx in range(self.graph_config[f"n_{self.mode}"]):
            instance = generate_solvable_instance_bounded_target(num_operands=num_operands, target_min=-target_abs, target_max=target_abs)
            nums = instance["inputs"]
            target = instance["target"]
            solution_trace = instance["solution_trace"]

            # Create a graph representation (placeholder, as actual graph structure is domain-specific)
            globals = {
                "numbers": np.array(nums),
                "target": np.array([target]),
                "solution_nodes": tokenize_rpn(num_operands, instance["solution_trace_rpn"]),
            }
            dummy_node_edge = np.array([0])

            n_nodes = num_operands * 2

            graph = GraphsTuple(np.zeros((n_nodes)), dummy_node_edge, dummy_node_edge, dummy_node_edge, globals, np.array([n_nodes]), np.array([1]), )

            # Store the instance and its solution
            solutions["H_graphs"].append(CountdownGraph(graph, {"solution_trace": solution_trace, "target_abs": target_abs, "op_abs": op_abs, "num_operands": num_operands}))
            solutions["Energies"].append(1e-6)  # Placeholder for energy
            solutions["gs_bins"].append(solution_trace)
            solutions["graph_sizes"].append(len(nums))
        return solutions
