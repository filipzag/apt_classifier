"""Lean attribution core: one ranker over dated report text, an optional proposer, one naming rule.

Design: ``/home/fzagar/.claude/plans/cheerful-meandering-puddle.md`` (2026-11-16). Kept apart from
``apt_classifier``'s engine on purpose: it imports the ATT&CK knowledge base, the knowledge store's
read side and the report anonymizer, never the benchmark module, torch or chromadb.
"""
