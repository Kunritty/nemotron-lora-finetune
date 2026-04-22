"""
Orchestration layer for agentic reasoning and data synthesis.

This package implements the 'Teacher-Student' distillation pipeline. It contains
the logical harnesses that enable high-reasoning models (Teachers) to solve 
complex puzzles and generate verified Chain-of-Thought (CoT) traces. These 
traces are subsequently used to fine-tune smaller adapters (Students).
"""
import memory
import prompts
import tools

__all__ = ["memory", "prompts", "tools"]