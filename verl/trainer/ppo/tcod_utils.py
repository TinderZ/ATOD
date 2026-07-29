"""
TCOD (Temporal Curriculum for On-Policy Distillation) utilities.

Implements curriculum scheduling strategies for multi-turn agent distillation:
  - f2b (Forward-to-Backward): Start from early steps, gradually extend window.
  - b2f (Backward-to-Forward): Start from late steps (with expert prefix), gradually
    reduce expert prefix length.

Reference: TCOD paper (arXiv:2604.24005)
"""


def compute_f2b_max_steps(
    global_step: int,
    checkpoint_steps: int,
    max_env_steps: int,
) -> int:
    """Compute effective max steps for TCOD-f2b.

    The student starts by only doing 1 step, and every `checkpoint_steps` training
    steps the window expands by 1.

    Args:
        global_step: Current training step.
        checkpoint_steps: How many training steps before expanding by 1 step.
        max_env_steps: Maximum environment steps (upper bound).

    Returns:
        effective_max_steps: How many steps the student should execute this iteration.
    """
    distill_window = 1 + (global_step // checkpoint_steps)
    return min(distill_window, max_env_steps)


def compute_b2f_expert_prefix_len(
    global_step: int,
    checkpoint_steps: int,
    total_expert_actions: int,
) -> int:
    """Compute expert prefix length for TCOD-b2f.

    The student starts by only doing the last step (expert does N-1 steps),
    and every `checkpoint_steps` training steps the expert prefix shrinks by 1.

    Args:
        global_step: Current training step.
        checkpoint_steps: How many training steps before reducing expert prefix by 1.
        total_expert_actions: Total number of expert/predefined actions available.

    Returns:
        expert_prefix_len: How many steps the expert should execute before student takes over.
            0 means student does everything from scratch.
    """
    if total_expert_actions <= 0:
        return 0
    max_expert_prefix = total_expert_actions - 1  # At least let student do 1 step
    reduction = global_step // checkpoint_steps
    expert_prefix_len = max(0, max_expert_prefix - reduction)
    return expert_prefix_len
