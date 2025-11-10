# YC Application - Agora

## 1. Describe what your company does in 50 characters or less.

**Orchestration framework for AI agent workflows**

(48 characters)

Alternative options:
- "Agent orchestration with built-in observability" (49 chars)
- "Workflow engine for multi-agent AI systems" (42 chars)

---

## 2. What is your company going to make? Please describe your product and what it does or will do.

**Agora is an orchestration framework for building multi-agent AI systems with production-grade observability.**

Developers building AI agents face two critical problems:
1. **Complex workflow orchestration** - Managing state, routing, retries, and error handling across multiple agents
2. **Zero visibility** - No way to debug, monitor, or understand what's happening inside agent workflows

Agora solves both:

**Simple Orchestration:**
```python
@agora_node(name="Analyst")
async def analyze_data(shared):
    result = await openai.call(shared["data"])
    shared["analysis"] = result
    return "summarize"  # Routes to next agent

flow = TracedAsyncFlow("Pipeline")
flow.start(analyze_data)
analyze_data - "summarize" >> summarizer_node
```

**Built-in Observability:**
Every node execution is automatically traced with OpenTelemetry. Developers get:
- Full execution traces (prep/exec/post phases)
- LLM call instrumentation (latency, tokens, cost)
- Agent routing decisions and state transitions
- Integration with any OpenTelemetry backend (Datadog, Honeycomb, etc.)

**Key differentiators:**
- **Decorator-based API** - Wrap existing functions without refactoring
- **Production telemetry** - OpenTelemetry standard, not proprietary logging
- **Stateful workflows** - Shared dict for cross-agent state management
- **Flexible routing** - Return values control agent transitions
- **Retry/fallback** - Built-in error handling at node level

**Target users:**
- AI engineers building multi-agent systems (RAG pipelines, research agents, coding assistants)
- Teams moving from prototypes to production (need observability + reliability)
- Enterprises requiring compliance/audit trails for AI systems

**Roadmap:**
- **Q1 2025:** Distributed execution (multiple machines)
- **Q2 2025:** Visual workflow builder + real-time debugging UI
- **Q3 2025:** Cost optimization engine (route to cheaper models when possible)
- **Q4 2025:** Pre-built agent marketplace (plug-and-play agents)

---

## 3. How far along are you?

**Status: Working prototype in production use**

**Current state:**
- ✅ Core orchestration engine (AsyncNode/AsyncFlow)
- ✅ Traceloop/OpenTelemetry integration
- ✅ @agora_node decorator for easy adoption
- ✅ Retry/fallback/error handling
- ✅ State management (shared dict)
- ✅ Routing system based on return values
- ✅ Python package (pip installable from GitHub)
- ✅ Working examples (chat apps, multi-agent pipelines)

**Early traction:**
- [Fill in: number of users/companies testing it]
- [Fill in: GitHub stars, if applicable]
- [Fill in: any production deployments]

**Revenue:**
- Not yet monetizing (open-source framework)
- Exploring enterprise support + managed platform

**What's missing for scale:**
- Distributed execution across machines
- Visual workflow builder UI
- Commercial-grade documentation
- SDKs for other languages (TypeScript, Go)

---

## 4. How long have each of you been working on this? How much of that has been full-time? Please explain.

[UPDATE THIS SECTION WITH YOUR ACTUAL TIMELINE]

**Example template:**

**[Founder Name]:**
- Working on Agora for [X months/years]
- [Y months] full-time, [Z months] part-time (while at [Previous Job/School])
- Background: [Relevant experience - e.g., "Built agent systems at [Company], saw these problems firsthand"]

**Context on how this started:**
- [Month/Year]: Started building internal tool for [use case]
- [Month/Year]: Realized orchestration + observability gap in market
- [Month/Year]: Open-sourced framework, got early users
- [Month/Year]: Went full-time after [traction metric / validation]

**Current status:**
- [X] founders working full-time
- [Y] contributors (if applicable)

---

## 5. What tech stack are you using, or planning to use, to build this product? Include AI models and AI coding tools you use.

**Current Stack:**

**Framework:**
- Python 3.9+ (async/await for concurrency)
- OpenTelemetry SDK (telemetry standard)
- Traceloop SDK (LLM instrumentation)

**Infrastructure:**
- Git/GitHub (version control + distribution)
- pip (package distribution)
- pytest (testing)

**AI Models Used:**
- OpenAI (GPT-4o-mini, GPT-4) - primary LLM for examples
- Support for any OpenAI-compatible API (Anthropic, local models, etc.)

**AI Coding Tools:**
- Claude Code (development assistance)
- GitHub Copilot (code completion)
- [Add any others you use]

**Planned Additions:**

**Backend (for managed platform):**
- FastAPI (agent execution API)
- PostgreSQL (workflow/trace storage)
- Redis (state management + caching)
- Temporal or Celery (distributed task execution)

**Frontend (workflow builder UI):**
- React + TypeScript
- React Flow (visual workflow editor)
- TanStack Query (data fetching)
- Tailwind CSS (styling)

**Observability:**
- Grafana (metrics dashboards)
- Honeycomb or Datadog (trace analysis)
- Prometheus (system metrics)

**Deployment:**
- Docker + Kubernetes (container orchestration)
- AWS/GCP (cloud infrastructure)
- Terraform (infrastructure as code)

---

## 6. Who are your competitors? What do you understand about your business that they don't?

**Competitors:**

**1. LangChain / LangGraph**
- Focus: General-purpose LLM framework
- Weakness: Observability is an afterthought (LangSmith is separate product). Complex API, steep learning curve.

**2. CrewAI**
- Focus: Multi-agent coordination with roles
- Weakness: Opinionated agent structure. Limited observability. Not designed for production scale.

**3. AutoGen (Microsoft)**
- Focus: Multi-agent conversations
- Weakness: Research-oriented, not production-ready. No built-in telemetry. Complex setup.

**4. Temporal / Prefect (workflow engines)**
- Focus: General task orchestration
- Weakness: Not AI-native. No LLM instrumentation. Overkill for simple agent workflows.

**5. LlamaIndex**
- Focus: RAG pipelines
- Weakness: Limited to RAG use cases. Not designed for general agent orchestration.

**What we understand that they don't:**

**1. Observability is non-negotiable, not a feature**
- Competitors treat observability as a premium add-on (LangSmith costs extra)
- We believe it should be built-in from day one - you can't debug agent systems without it
- Using OpenTelemetry (not proprietary format) means teams can use their existing tools

**2. Developers want simplicity, not frameworks**
- LangChain has 1000+ classes. Developers want to write functions, not learn a DSL.
- Our decorator approach: wrap existing code, minimal refactoring
- "Framework" vs "library" - we're a library that stays out of your way

**3. The market is enterprises with existing agents, not researchers**
- Competitors target AI researchers building new things from scratch
- Real market: Companies moving prototypes to production (need reliability + observability)
- They already have working agent code - they need orchestration + telemetry, not a rewrite

**4. Agent workflows are just DAGs with state**
- Temporal/Prefect are too complex for this
- LangChain is too opinionated (chains, prompts, etc.)
- Simple node->action->node routing + shared state solves 90% of cases

**5. Production-readiness requires standard protocols**
- OpenTelemetry is the standard (not vendor-specific logging)
- Python async/await for concurrency (not threads/multiprocessing)
- Pip installable (not "deploy our infrastructure")

**Our unfair advantage:**
- We've built agent systems in production and felt these pain points firsthand
- We're not building a research project - we're solving a real operational problem
- Simple API (decorators) + production telemetry (OpenTelemetry) is a unique combination

---

## 7. How do or will you make money? How much could you make?

**Business Model: Open-Core**

**Phase 1: Open Source Framework (Current)**
- Core orchestration engine is free and open-source
- Build community, establish standard
- No revenue (investing in adoption)

**Phase 2: Managed Platform (6-12 months)**
- Agora Cloud: Hosted execution + observability backend
- Pricing: Usage-based
  - $0.01 per 1000 node executions
  - $50/mo per developer for trace retention + analysis
  - $500-5000/mo for enterprise features (SSO, compliance, SLAs)

**Phase 3: Enterprise Support (12-18 months)**
- Self-hosted enterprise edition
- Features: Private deployment, advanced security, custom integrations
- Pricing: $50k-500k/year per organization

**Revenue Potential:**

**Market size:**
- AI agent development is exploding (every company building agents)
- Comparable to observability market (Datadog: $2B revenue, Honeycomb: $100M+)
- Workflow orchestration market (Temporal: $1B+ valuation)

**Target customers:**
- Startups building AI products (1000s of companies)
- Mid-market (AI teams at Series A-C companies)
- Enterprises (Fortune 500 building internal agent systems)

**Unit economics (example):**
- Startup (10 developers): $500-2000/mo = $6k-24k/year
- Mid-market (50 developers): $2500-10k/mo = $30k-120k/year
- Enterprise (500+ developers): $50k-500k/year

**Conservative projections:**
- Year 1: 10 paying customers → $100k ARR (validation)
- Year 2: 100 paying customers → $1M ARR (product-market fit)
- Year 3: 500 paying customers → $10M ARR (scale)
- Year 5: 2000+ customers → $50M+ ARR (market leader)

**Why this works:**
- **Low friction:** Start free (open-source), upgrade when you scale
- **Must-have product:** Can't run agents in production without observability
- **Sticky:** Once integrated into workflows, hard to switch
- **Expands with usage:** More agents/executions → more revenue

**Comparable exits:**
- Temporal: $1B+ valuation (workflow orchestration)
- Honeycomb: $100M+ revenue (observability)
- Datadog: $40B market cap (observability at scale)

If we capture even 1% of the agent orchestration market, that's a $100M+ business.

---

## NOTES FOR APPLICATION:

**Key messages to emphasize:**
1. **Real problem:** Agent systems are black boxes - debugging is impossible
2. **Simple solution:** Decorator-based API + built-in observability
3. **Production focus:** Not a research project - solving real operational pain
4. **Timing:** Every company is building agents NOW - perfect market moment
5. **Traction:** [Fill in with actual usage numbers, GitHub activity, etc.]

**Questions they'll ask:**
- "Why not just use LangChain?" → Too complex, observability is separate product
- "Why OpenTelemetry?" → Industry standard, works with existing tools
- "Who pays?" → Companies moving prototypes to production (need reliability)
- "Why you?" → [Fill in: we've built this in production, felt the pain]

**Red flags to address:**
- Open-source monetization → We're open-core (hosted platform + enterprise)
- Competition → We're focused on production use cases, not research
- Technical moat → Developer experience + OpenTelemetry integration is hard to replicate

---

**NEXT STEPS:**
1. Fill in actual timeline (section 4)
2. Add traction metrics (section 3)
3. Add founder background (section 4)
4. Quantify early users/adoption (section 3)
5. Add any production deployments as proof points
