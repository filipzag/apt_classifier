"""Dated knowledge store: public research posts read into dated, linked records.

Design: docs/KNOWLEDGE_STORE_DESIGN.md. Step 1 (this package today) lists the approved sources
(``sources``), downloads politely (``fetch``), reads each post (``parse``), dates it
(``listing``) and keeps it in SQLite (``store``); ``scripts/knowledge_store.py`` runs it. Every
consumer reads through ``KnowledgeStore.view(as_of)``, so nothing published on or after a case's
date is used. The text belongs to each publisher: the store is local and never redistributed.
"""
