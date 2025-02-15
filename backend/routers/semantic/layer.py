from langchain_huggingface import HuggingFaceEmbeddings
from semantic_router import Route, RouteLayer
from semantic_router.encoders import HuggingFaceEncoder
# from sentence_transformers import SentenceTransformer
# from .encoder import get_encoder

class SemanticRouter:
    def __init__(self):
        self.encoder = HuggingFaceEncoder(name='sentence-transformers/all-MiniLM-L6-v2') #SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') #get_encoder()
        self.routes = self._load_routes()
        self.layer = RouteLayer(
            encoder=self.encoder,
            routes=self.routes
        )
    
    def route(self, query: str):
        return self.layer(query)
    
    def _load_routes(self):
        return [
            Route(
                name="web_search",
                utterances=[
                    "search for",
                    "find information about",
                    "look up"
                ],
                handler_fn="handlers.web_search.process"
            ),
            Route(
                name="coding",
                utterances=[
                    "how to code",
                    "programming question",
                    "debug this"
                ],
                handler_fn="handlers.coding.process"
            ),
            Route(
                name="profile_search",
                utterances=[
                    "which profile",
                    "i'm looking for job profile",
                ],
                handler_fn="handlers.coding.process"
            )
        ]

route_layer = SemanticRouter()
