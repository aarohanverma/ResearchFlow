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

.. toctree::
   :maxdepth: 3
   :caption: Workflows — Media Generation

   api/workflows._generation_runtime
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
