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
        return self.layer(query[:400])
    
    def _load_routes(self):
        return [
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
                    "I need a profile for a nurse using accountabilities from band 1 terms",
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
                    "what is the classification for this profile",
                    "can you classify this profile",
                    "can you classify this profile: "
                ],
                handler_fn="handlers.coding.process"
            ),
            Route(
                name="provide_help",
                utterances=[
                    "tell me what i can do in jobstore",
                    "what can i do in jobstore?",
                    "what are jobstore's capabilities?",
                    "how do i access jobstore?",
                    "can i create multiple positions at once?",
                    "is it possible to download org charts?",
                    "does the system create position numbers automatically?",
                    "can i duplicate existing positions?",
                    "what job profiles are available?",
                    "can i edit job profiles?",
                    "how do i download a job profile?",
                    "who has access to jobstore?",
                    "what access do people leaders have?",
                    "what's the difference between excluded and included access?",
                    "do i need special permissions?",
                    "can employees access jobstore?",
                    "how do i get access to jobstore?",
                    "what happens after i submit for verification?",
                    "does it create service requests automatically?",
                    "how long to get a position number?",
                    "what does verification required mean?",
                    "do i need separate service requests?",
                    "what happens during an audit?",
                    "how does classification work?",
                    "is jobstore connected to peoplesoft?",
                    "what if the org chart is wrong?",
                    "how do i fix technical issues?",
                    "what file formats are available for downloads?",
                    "where is the download button?",
                    "what should i do if i get an error?",
                    "what are jobstore's limitations?",
                    "why can't i create certain positions?",
                    "are there restrictions on excluded positions?",
                    "can i create positions without approved profiles?",
                    "why can't i start a new org chart?",
                    "who do i contact for support?",
                    "what if i can't find the job profile i need?",
                    "how do i report issues?",
                    "where can i find help docs?",
                    "can i make changes after creating a position?",
                    "who handles classification questions?"
                ],
                handler_fn="handlers.coding.process"
            ),
            Route(
                name="provide_self_help",
                utterances=[
                    "i need help",
                    "tell me about yourself",
                    "hello",
                    "what can I do?",
                    "What can you do?",
                    "What are your main features?",
                    "How do you work?",
                    "What kind of tasks can you help me with?",
                    "Tell me about your capabilities",
                    "What functions do you support?",
                    "How can you help me with job profiles?",
                    "What are the different ways I can use this system?",
                    "Give me an overview of your features",
                    "What are your core functionalities?",
                    "What backend architecture do you use?"
                    "How does your semantic routing work?",
                    "What embedding model do you use?",
                    "How do you process and store data?",
                    "What's your classification system based on?",
                    "How do you handle vector searches?",
                    "What programming language are you built with?",
                    "How do you integrate with FastAPI?",
                    "What kind of database do you use?",
                    "How do you maintain data consistency?",
                ],
                handler_fn="handlers.coding.process"
            ),

            Route(
                name="not_supported",
                utterances=[
                    "Generate a concise, 3-5 word title with an emoji summarizing the chat history",
                ],
                handler_fn="handlers.coding.process"
            )
            
        ]

route_layer = SemanticRouter()
