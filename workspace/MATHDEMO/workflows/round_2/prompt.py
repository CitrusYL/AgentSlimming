CONCISE_SOLUTION_PROMPT = """
Solve the math problem carefully and return a concise solution.

Requirements:
1. Show the core reasoning steps only.
2. Keep the derivation short and correct.
3. End with the final answer in \\boxed{} notation.
"""


DETAILED_SOLUTION_PROMPT = """
Solve the math problem step by step.

Requirements:
1. Explain the main ideas clearly.
2. Show the intermediate derivation in enough detail to verify correctness.
3. End with the final answer in \\boxed{} notation.
"""


ALTERNATIVE_SOLUTION_PROMPT = """
Solve the math problem using an alternative approach from the most obvious one.

Requirements:
1. Prefer a different derivation or observation if possible.
2. Keep the reasoning readable and mathematically sound.
3. End with the final answer in \\boxed{} notation.
"""
