ResearchFlow — API Reference
============================

**ResearchFlow** is an AI-native research operating system. This reference covers every
module, class, and function in the backend.

.. toctree::
   :maxdepth: 1
   :caption: Overview

   overview

.. toctree::
   :maxdepth: 3
   :caption: Workflows — Core

   api/workflows.ingestion
   api/workflows.study
   api/workflows.rag
   api/workflows.genie
   api/workflows.genie_combine
   api/workflows.folder_consolidation

.. toctree::
   :maxdepth: 3
   :caption: Workflows — Media Generation

   api/workflows._generation_runtime
   api/workflows._generation_prompts
   api/workflows.podcast
   api/workflows.slides

.. toctree::
   :maxdepth: 3
   :caption: API Routers

   api/api.v1.auth
   api/api.v1.feed
   api/api.v1.papers
   api/api.v1.study
   api/api.v1.chat
   api/api.v1.search
   api/api.v1.genie
   api/api.v1.graph
   api/api.v1.bookmarks
   api/api.v1.settings
   api/api.v1.generate
   api/api.v1.assistant
   api/api.v1.admin
   api/api.v1.dev

.. toctree::
   :maxdepth: 3
   :caption: Research Assistant — Orchestration

   api/assistant.orchestrator
   api/assistant.planner
   api/assistant.planner_llm
   api/assistant.synthesizer
   api/assistant.events
   api/assistant.scheduler
   api/assistant.recovery
   api/assistant.step_cache
   api/assistant.session_metadata
   api/assistant.interest_updater
   api/assistant.intent
   api/assistant.persona
   api/assistant.prompt_safety
   api/assistant.query_strategy
   api/assistant.research_brief
   api/assistant.branch_context
   api/assistant.state_lock
   api/assistant.telemetry
   api/assistant.tuning
   api/assistant.clarify
   api/assistant.reflection
   api/assistant.repair_drift
   api/assistant.provenance
   api/assistant.auto_memory
   api/assistant.semantic_memory
   api/assistant.memory_consolidation

.. toctree::
   :maxdepth: 3
   :caption: Research Assistant — ReAct loop & middlewares

   api/assistant.react_loop
   api/assistant.react
   api/assistant.react.middlewares
   api/assistant.scratchpad
   api/assistant.retrieval_observability
   api/assistant.contradiction
   api/assistant.claim_ledger
   api/assistant.hitl_inbox

.. toctree::
   :maxdepth: 3
   :caption: Research Assistant — Tools

   api/assistant.tools.base
   api/assistant.tools.registry

.. toctree::
   :maxdepth: 3
   :caption: Repositories

   api/repositories.paper
   api/repositories.vector
   api/repositories.search
   api/repositories.graph
   api/repositories.user
   api/repositories.workflow
   api/repositories.artifact
   api/repositories.assistant

.. toctree::
   :maxdepth: 3
   :caption: Services

   api/services.scoring
   api/services.graph
   api/services.namespace
   api/services.token_usage
   api/services.email_service
   api/services.content_loader
   api/services.job_store
   api/services.research_assistant
   api/services.arxiv_import
   api/services.semantic_chunker
   api/services.feature_flags
   api/services.admin_settings

.. toctree::
   :maxdepth: 3
   :caption: Adapters

   api/adapters.llm
   api/adapters.embedding
   api/adapters.pdf
   api/adapters.cache
   api/adapters.blob
   api/adapters.image_gen
   api/adapters.sources
   api/adapters.email
   api/adapters.web_search
   api/adapters.tts
   api/adapters.slides

.. toctree::
   :maxdepth: 3
   :caption: Models

   api/models.paper
   api/models.genie
   api/models.graph
   api/models.user
   api/models.workflow
   api/models.artifact
   api/models.assistant
   api/models.admin
   api/models.rbac

.. toctree::
   :maxdepth: 2
   :caption: Core

   api/core.config
   api/core.security
   api/core.deps
   api/schemas
   api/db
   api/resilience
   api/scheduler
   api/tools

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
