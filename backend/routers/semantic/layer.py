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
        return self.layer(query[:250])
    
    def _load_routes(self):
        return [
            Route(
                name="profile_search",
                utterances=[
                    "which profile",
                    "which profiles",
                    "i'm looking for job profile",
                    "find profiles related to project management",
"search for data scientist job profiles",
"show me profiles that require python skills",
"which profiles include leadership responsibilities",
"find profiles with budget management requirements",
"search profiles mentioning agile methodology",
"what profiles exist for technical writers",
"show profiles requiring security clearance",
"find profiles with stakeholder engagement",
"which profiles involve cloud architecture",
"search for profiles with similar responsibilities to",
"find profiles that mention machine learning",
"show me all profiles in finance department",
"which profiles require MBA qualification",
"find profiles with remote work options",
"search for profiles with team lead responsibilities",
"show profiles requiring specific certifications",
"find profiles similar to business analyst",
"what profiles exist for entry level positions",
"search profiles by required years of experience",
"show me profiles with research responsibilities",
"find profiles requiring specific software knowledge",
"which profiles involve data analysis",
"search for profiles with client-facing roles",
"find profiles mentioning specific technologies",

"tell me what skills appear in",
"analyze skills mentioned across",
"compare requirements between",
"list common qualifications in",
"extract skills from",
"identify patterns in",
"summarize requirements from",
"what do the profiles say about",
"show me trends in",
"find common elements in",
"what requirements appear in",
"how many profiles mention",
"analyze similarities between",
"what skills overlap in",
"examine patterns across",
"Which profiles have software development qualities?",
"which jobs are most fun",
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
                    "create a profile combining data science and management skills",
"draft a new profile for cloud security specialist",
"build a profile for senior UX researcher",
"generate hybrid profile for devops engineer",
"create custom profile with these requirements",
"write new profile for product owner role",
"develop profile for AI ethics officer",
"construct profile for sustainability manager",
"synthesize profile for digital transformation lead",
"create profile based on these competencies",
"build new profile with following qualifications",
"generate profile using these key responsibilities",
"draft comprehensive profile for systems architect",
"create profile merging these two roles",
"build profile with emphasis on innovation",
"generate profile incorporating these skills",
"create profile with specific education requirements",
"develop new profile for emerging tech role",
"construct profile with these reporting relationships",
"generate profile based on industry standards",
"create profile adapting these requirements",
"build specialized profile for research position",
"draft profile with focus on leadership",
"generate profile with specific technical stack",
"create hybrid role profile combining",
                ],
                handler_fn="handlers.coding.process"
            ),
            Route(
                name="draw_profiles_graph",
                utterances=[
                    "make a piechart of the number of profiles by organization",
                    "make a pie chart of profiles counts broken down by job family",
                    "show job profile graph",
                    "visualize job profiles",
                    "create profile classification chart",
                    "graph profile views",
                    "display profile statistics",
                    "plot profile trends",
                    "analyze job profiles",
                    "profile type distribution",
                    "show profile classifications",
                    "job profile dashboard",
                    "profile view metrics",
                    "show job profile views by role type",
                    "graph corporate vs operational profiles",
                    "show profile distribution by classification",
                    "visualize job profile views over time",
                    "show profiles by job family",
                    "plot profiles by stream",
                    "show individual contributor profile distribution",
                    "graph profiles by organization scope",
                    "show most viewed job profiles",
                    "visualize profiles by valid date range",
                    "show operational administration profiles",
                    "graph profiles with reports to relationships",
                    "show profile distribution by scope",
                    "plot profile creation timeline",
                ],
                handler_fn="handlers.coding.process"
            ),
            Route(
                name="draw_pr_graph",
                utterances=[
                     "show position request graph",
                    "visualize position requests",
                    "create position request chart",
                    "graph position request data",
                    "display position request statistics",
                    "plot position request trends",
                    "analyze position requests",
                    "position request distribution",
                    "show request approval status",
                    "position request dashboard",
                    "position approval visualization",
                    "show approval status distribution for position requests",
                    "graph top 5 requested job profiles",
                    "show position request approvals over time",
                    "visualize employee group distribution in requests",
                    "show classification grade distribution for requests",
                    "plot department code versus organization for requests",
                    "show requests by classification code",
                    "graph position requests by approval type",
                    "show BCGEU position request trends",
                    "visualize requests by peoplesoft ID",
                    "show completed vs pending position requests",
                    "graph position requests by organization name",
                    "show verified vs unverified request distribution",
                    "plot position requests by department",
                ],
                handler_fn="handlers.coding.process"
            ),
            Route(
                name="classify_profile",
                utterances=[
                    "classify this profile",
                    "what is the classification for this profile",
                    "can you classify this profile",
                    "can you classify this profile: ",
                    "Can you classify this profile?",
                    "determine the classification of this profile",
"evaluate profile classification level",
"assess this profile's classification",
"find appropriate classification grade",
"suggest classification for this role",
"analyze profile for classification level",
"review and classify this job description",
"provide classification assessment",
"identify classification category",
"recommend classification level",
"determine grade level for this profile",
"what classification band applies here",
"evaluate classification band for profile",
"assign classification to this profile",
"check classification category",
"examine profile for classification",
"calculate classification rating",
"establish profile classification",
"derive classification from profile",
"indicate classification level",
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
