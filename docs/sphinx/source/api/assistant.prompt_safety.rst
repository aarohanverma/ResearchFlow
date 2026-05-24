assistant — prompt safety wrappers
====================================

Helpers that wrap untrusted paper / web content in delimiter markers so
the LLM treats it as DATA rather than instructions. Used by every tool
that injects retrieved text into a prompt and by the synthesizer.

.. automodule:: app.assistant.prompt_safety
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
