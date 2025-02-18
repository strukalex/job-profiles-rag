## Drag graph examples

Exactly matches examples:
- `Show top 5 organizations by total views`
- `Response for "Show views by role type"`

Novel (simple):
- `Show views by type`
- `Show views by job family (top 5)`
- `Show views distribution by organizational type`
- `show graph of profile creation by month`
- `show graph of profile creation by month, add readable month names`
- `Show graph of the number of profiles broken down by classification, ordered from highest to lowest, remove labels for classifications (y-axis)`
- `make a pie chart of the number of profiles by organization`
- `make a pie chart of the number of profiles by organization, use rainbow theme, include percentage values`
- `make a pie chart of profiles counts broken down by job family apply normal color scheme, include percentages`

## Vector search demonstration

```
# orig: 
# text="Gains cooperation and consensus on cloud-based solutions among a diverse group of managers and facilitates the development of migration plans."

# slight variants:
# "Can you generate several variants of this statement that mean the same thing, but just worded very slightly differently?"

# text="Builds consensus and collaboration with diverse management teams to develop cloud migration strategies and implementation plans."
# match:  Gains cooperation and consensus on cloud-based solutions among a diverse group of managers and facilitates the development of migration plans.
# threshold:  0.28416338562965393

# text="Facilitates agreement among varied management stakeholders on cloud solutions while coordinating migration planning efforts."
# match:  Gains cooperation and consensus on cloud-based solutions among a diverse group of managers and facilitates the development of migration plans.
# threshold:  0.3001526892185211

# (0.249) text="Drives cooperative decision-making on cloud initiatives across diverse management groups and guides migration planning processes."
# match:  Gains cooperation and consensus on cloud-based solutions among a diverse group of managers and facilitates the development of migration plans.
# threshold:  0.24942871928215027


# "can you now make it more dissimilar, but still related?"

# text="Leads enterprise-wide cloud transformation initiatives by unifying cross-functional leadership perspectives and architecting transition strategies."
# match:  Develops on-premise to cloud transformation plans to determine roadmaps to transition desktop software titles to SaaS.
# threshold:  0.8024989366531372

# text="Champions organizational cloud adoption through stakeholder engagement and strategic migration orchestration."
# match:  Gains cooperation and consensus on cloud-based solutions among a diverse group of managers and facilitates the development of migration plans.
# threshold:  0.7313711047172546

# text="Bridges technical and business objectives by aligning leadership teams on cloud modernization efforts and execution frameworks.""
# match:  Gains cooperation and consensus on cloud-based solutions among a diverse group of managers and facilitates the development of migration plans.
# threshold:  0.8453207612037659


# now can you generate some that are in between the two lists you made in terms of similarity?

# text="Aligns diverse management perspectives on cloud transformation initiatives while developing structured migration approaches."
# match:  Gains cooperation and consensus on cloud-based solutions among a diverse group of managers and facilitates the development of migration plans.
# threshold:  0.5251539349555969

# text="Coordinates cross-functional management consensus for cloud solutions and establishes pragmatic migration frameworks."
# match:  Gains cooperation and consensus on cloud-based solutions among a diverse group of managers and facilitates the development of migration plans.
# threshold:  0.6402548551559448

# text="Facilitates management alignment on cloud strategies while guiding the organization through migration planning and execution."
# match:  Gains cooperation and consensus on cloud-based solutions among a diverse group of managers and facilitates the development of migration plans.
# threshold:  0.5055486559867859

# dissimilar (example generated)
# (1.022) text="Designing and implementing scalable, high-performance cloud-based applications and services"
# match:  Builds and contributes to logging, monitoring, and alerting systems to identify bottlenecks and assist with debugging, analysis, and optimization in a cloud-agnostic environment.
# threshold:  1.0220295190811157
```