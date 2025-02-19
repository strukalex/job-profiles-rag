from semantic_router import Route, RouteLayer
from semantic_router.encoders import HuggingFaceEncoder

class SemanticRouter:
    def __init__(self):
        self.encoder = HuggingFaceEncoder(name='thenlper/gte-small')
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
                    "i need a profile for software engineer",
                    "i need a new job profile",
                    "I need a profile for a nurse using accountabilities from band 1 classification terms",
                    "make a job profile for",
                ],
                handler_fn="handlers.coding.process"
            ),
            Route(
                name="draw_graph",
                utterances=[
                    "make a chart of",
                    "generate a graph",
                    "make a graph",
                    "make a graph of a number of",
                    "Show top 5 organizations by total views",
                    "Show views by role type",
                    "make a piechart of the number of profiles by organization",
                    "make a pie chart of profiles counts broken down by job family"
                ],
                handler_fn="handlers.coding.process"
            ),
            Route(
                name="classify_profile",
                utterances=[
                    "classify this profile",
                    "what is the classification for this profile"
                ],
                handler_fn="handlers.coding.process"
            )
        ]

route_layer = SemanticRouter()
