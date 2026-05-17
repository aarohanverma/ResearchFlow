"""Mount all v1 sub-routers."""

from fastapi import APIRouter

from app.api.v1 import assistant, auth, bookmarks, chat, dev, feed, genie, generate, graph, papers, search, settings, study

router = APIRouter(prefix="/api/v1")

router.include_router(auth.router)
router.include_router(assistant.router)
router.include_router(feed.router)
router.include_router(search.router)
router.include_router(papers.router)
router.include_router(study.router)
router.include_router(bookmarks.router)
router.include_router(graph.router)
router.include_router(chat.router)
router.include_router(genie.router)
router.include_router(settings.router)
router.include_router(generate.router)
router.include_router(dev.router)
