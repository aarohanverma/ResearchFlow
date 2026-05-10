"""Import all models so SQLAlchemy registers them on Base.metadata."""

from app.models.genie import GenieElement, GenieSession, IdeaCapsule
from app.models.graph import (
    KnowledgeEdge,
    KnowledgeNode,
    NamespaceSubscription,
    SourceMapping,
)
from app.models.paper import (
    Bookmark,
    BookmarkFolder,
    BookmarkFolderMember,
    FeedFeedback,
    Paper,
    PaperChunk,
    PaperCitation,
    PaperOfDay,
    QueryLog,
    Summary,
)
from app.models.user import Annotation, User, UserInterestProfile, UserProviderSettings
from app.models.workflow import TokenUsage, WorkflowRun
from app.models.artifact import GeneratedArtifact

__all__ = [
    "User", "UserProviderSettings", "UserInterestProfile", "Annotation",
    "Paper", "PaperChunk", "Summary", "Bookmark", "BookmarkFolder",
    "BookmarkFolderMember", "PaperOfDay", "PaperCitation", "QueryLog", "FeedFeedback",
    "KnowledgeNode", "KnowledgeEdge", "NamespaceSubscription", "SourceMapping",
    "WorkflowRun", "TokenUsage",
    "GenieElement", "IdeaCapsule", "GenieSession",
    "GeneratedArtifact",
]
