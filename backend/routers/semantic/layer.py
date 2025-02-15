from semantic_router import Route, RouteLayer
from semantic_router.encoders import HuggingFaceEncoder

class SemanticRouter:
    def __init__(self):
        self.encoder = HuggingFaceEncoder(name='sentence-transformers/all-MiniLM-L6-v2')
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
                    "which profiles",
                    "i'm looking for job profile",
                ],
                handler_fn="handlers.coding.process"
            ),
            Route(
                name="generate_profile",
                utterances=[
                    "generate profile",
                    "make new profile",
                    "i need a profile for software engineer"
                ],
                handler_fn="handlers.coding.process"
            )
        ]

route_layer = SemanticRouter()
