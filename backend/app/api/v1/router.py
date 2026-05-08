"""Mount all v1 sub-routers."""

from fastapi import APIRouter

from app.api.v1 import auth, bookmarks, chat, feed, genie, graph, papers, search, settings, study

router = APIRouter(prefix="/api/v1")

router.include_router(auth.router)
router.include_router(feed.router)
router.include_router(search.router)
router.include_router(papers.router)
router.include_router(study.router)
router.include_router(bookmarks.router)
router.include_router(graph.router)
router.include_router(chat.router)
router.include_router(genie.router)
router.include_router(settings.router)
