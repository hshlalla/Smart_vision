# CM3070 Past Exam Answer Bank

## 0. Core project summary

These notes are written for the project **Smart Image Part Identifier for Secondhand Platforms**.
The project follows the **CM3020 Artificial Intelligence, "Orchestrating AI models to achieve a goal"** template.
Its main idea is that identifying industrial or electronic parts from secondhand photos should be treated as a **retrieval-first, human-in-the-loop decision-support problem**, not as a closed-set classification problem.

The system combines:

- image retrieval
- OCR-derived text evidence
- text and metadata retrieval
- caption-based retrieval
- Milvus vector storage
- catalog retrieval over PDF documents
- an agent layer that can orchestrate tools and expose evidence

The most honest description of the current outcome is:

> a technically challenging end-to-end prototype for secondhand part identification assistance, with partial but meaningful evaluation evidence.

## Reusable points for many questions

### Strengths

- Strong problem fit: open-world, fine-grained, text-sensitive part identification is a real marketplace problem.
- Clear system contribution: integrates OCR, multimodal retrieval, catalog search, and agent orchestration.
- Honest design framing: shortlist + evidence + user verification rather than overclaiming automation.
- Working architecture across web, API, and model layers.
- Recent hardening work improved safety and testability.

### Limitations

- OCR remains brittle under glare, blur, small text, and occlusion.
- Full accept/edit/reject review workflow is not yet complete.
- CER/WER aggregation, latency percentiles, and full ablation studies are not yet fully completed.
- Current evaluation is meaningful but incomplete for a final production claim.

### Technical challenge shorthand

The hardest technical problem was not simply “doing OCR” or “doing image search,” but reliably
reading highly variable labels and combining that imperfect OCR/VLM evidence with body-image signals
to rank the right candidates. Document-oriented OCR was not naturally optimised for these labels,
VLM improved many cases but still struggled with damaged or erased markings, and weighted fusion of
multiple models created both calibration and latency trade-offs.


### Evaluation shorthand

Current evaluation evidence includes:

- retrieval-first qualitative and baseline evidence
- regression evidence from API and model pytest runs
- latency instrumentation in the hybrid search path
- critical discussion of limitations and next steps

What is still incomplete:

- aggregate CER benchmark
- p50/p90/p95 latency summaries
- large controlled hybrid ablation
- user study rerun

---

# 1. CM3070 Exam Mar 2025

## Q1. Project process

### (a) What are the four phases (the four “D’s”) of doing a project?

A clear way to describe the four D's is:

1. **Discover**: understand the problem, context, users, domain constraints, and relevant prior work.
2. **Design**: translate findings into requirements, architecture, evaluation criteria, and implementation plans.
3. **Develop**: build the prototype, test components, iterate on failures, and integrate the system.
4. **Discuss**: evaluate the outcome critically, interpret results, identify limitations, and communicate contributions and future work.

For this project, these phases map naturally onto the final report structure. Discovery informed the problem framing and literature review; design captured the retrieval-first architecture and evaluation strategy; development produced the web/API/model system; and discussion appears in the evaluation and conclusion chapters.

### (b) For each phase, describe your process for your own project.

**Discover.** I began by examining the secondhand listing problem for industrial and electronic parts. I identified that the domain is open-world, visually fine-grained, and often depends on tiny textual identifiers such as model codes or part numbers. I also reviewed prior work in image retrieval, OCR, multimodal embeddings, vector search, and marketplace assistance tools. This phase led me to reject a simple closed-set classifier as the main solution.

**Design.** I converted the problem analysis into concrete requirements: the system should support image-first search, remain useful when OCR fails, provide Top-K shortlist outputs, and expose evidence for verification. I designed a modular architecture with a React frontend, FastAPI backend, model package, hybrid search pipeline, Milvus vector collections, catalog retrieval, and an agent layer. I also treated evaluation as part of the design by planning retrieval, OCR, latency, and engineering-stability evidence.

**Develop.** I implemented the end-to-end workflow across web, API, and model layers. The search flow supports image upload and optional text queries, hybrid retrieval, and evidence-backed result display. I later expanded the system with catalog retrieval and agent orchestration. During the final phase I also improved reliability by making writeback opt-in by default, adding tests, and instrumenting latency-related stages.

**Discuss.** I critically reviewed what the prototype actually demonstrates. I concluded that the project succeeds as a retrieval-first assistant, but not yet as a fully autonomous identifier. I explicitly separated already supported claims from partial or future evaluation work. If I repeated the project, I would preserve the retrieval-first framing but would schedule quantitative evaluation automation much earlier.

## Q2. Presentation of the Project

### (a) How did you structure the final submission presentation of your project, namely the report and the video?

I structured the report around six chapters: introduction, literature review, design, implementation, evaluation, and conclusion. The key decision was to make the report tell a coherent story: why this problem is difficult, why a retrieval-first approach is appropriate, how the architecture addresses the problem, and what evidence currently supports the resulting claims. I also deliberately reframed the system as a **human-in-the-loop assistant** rather than an autonomous classifier, because that wording better fits the available evidence.

For the video, the most important choice would be to show the system working rather than only talking about architecture. I would focus on the web flow: uploading an image, retrieving a shortlist, showing evidence-aware results, and demonstrating the agent or catalog path briefly. This is effective because the marker can quickly see that the project is real, integrated, and usable.

A second key presentation choice is omission. I would avoid overloading the report or video with every implementation detail. Instead, I would prioritise the architecture, hybrid retrieval rationale, safety decisions, and evaluation honesty. This is more effective than trying to present every module equally.

### (b) Imagine you have to give a half-hour talk about your project.

An appropriate setting would be an internal product and research review at a marketplace or industrial e-commerce company. This is suitable because the project addresses a practical workflow problem: helping non-expert users identify technical parts for listing creation.

My talk would have around 14 to 16 slides:

1. problem and motivation
2. why secondhand part identification is hard
3. project aims and requirements
4. why closed-set classification is not enough
5. retrieval-first system overview
6. hybrid search pipeline
7. OCR and metadata fusion
8. architecture across web/API/model layers
9. catalog retrieval and agent orchestration
10. demo screenshots or short live demo
11. evaluation strategy
12. what evidence currently exists
13. limitations and risks
14. future work and impact

I would include architecture diagrams, one UI screenshot, one retrieval-evidence example, and one slide clearly separating supported claims from in-progress work. I would leave out low-level code details unless asked in Q&A, because the main goal of the talk is to communicate the problem, the system idea, and the evidence-based contribution.

## Q3. Reflection and continuation

### (a) Evaluate your project as a whole.

The project’s main strengths are its problem fit, systems integration, and honest framing. It addresses a real and difficult problem where user photos, OCR noise, open-world inventory, and listing-oriented outputs all matter. The strongest contribution is not a novel foundation model but a practical orchestration of OCR, multimodal retrieval, vector search, catalog lookup, and agent tooling into a usable workflow. Another strength is that the design now reflects uncertainty: results are presented as shortlist evidence for review rather than as overconfident final answers.

The main weaknesses concern evaluation completeness and workflow maturity. OCR is still brittle under realistic image conditions. The accept/edit/reject review flow is not yet fully developed across the user-facing path. Quantitative evaluation is also incomplete in important areas such as aggregate CER, latency percentiles, and controlled ablation between retrieval variants. These weaknesses do not invalidate the prototype, but they do limit the strength of the claims that can responsibly be made.

### (b) Imagine a future student extends your project.

A strong follow-on project would be **human-reviewed listing completion with audited feedback loops**. This would build on my current output by turning retrieval results into a more complete workflow where users can explicitly accept, edit, reject, and store corrected metadata. The motivation is strong because the current project already shows that retrieval and evidence are useful, but it stops short of a complete review-driven production loop.

A second strong follow-on direction would be **controlled multimodal ablation and region-focused retrieval**. This project would compare Qwen-centred and mixed OCR-plus-BGE pipelines, and optionally test region-focused OCR or object cropping for difficult cases. That would deepen the scientific evaluation of which signals matter most in fine-grained part identification.

To take either direction forward, I would advise the next student to preserve the current modular boundary between frontend, API, and model package, and to define evaluation criteria before implementing new features. The biggest pitfall would be adding complexity faster than evidence can be collected.

---

# 2. CM3070 Past Exam 2024-09

## Q1. Project management and execution

### (a) Primary objectives in the planning phase

My primary objectives were:

- to build a working end-to-end prototype for photo-based part identification
- to support open-world retrieval rather than closed-set classification
- to combine image and text evidence rather than depend on vision alone
- to produce outputs useful for listing workflows, not only raw similarity scores
- to leave enough instrumentation and evidence to evaluate the project honestly

These objectives were realistic because they focused on system integration and user assistance rather than trying to invent a new foundation model.

### (b) How I organised the initial stages

I organised the early stage around four workstreams: literature/problem framing, architecture design, prototype implementation, and evaluation planning. The key resources were the project template, previous report feedback, the selected model/database stack, and the datasets and images used for indexing and testing. Major tasks included clarifying requirements, choosing the retrieval-first architecture, defining API routes and UI flows, implementing indexing/search, and planning evidence collection.

The timeline was broadly iterative. I did not treat planning as something finished before implementation. Instead, design and development informed each other, especially when OCR reliability and open-world retrieval constraints became clearer.

### (c) Analyse effectiveness of the initial project plan

The initial plan worked better than I expected. The core problem definition stabilised relatively
early, and the main planned features were achieved faster than I had anticipated. That gave me room
later to add extensions such as the agent path and the catalog path.

However, in hindsight, the main weakness was not failure to meet the original aims but difficulty in
controlling scope after those aims were reached. Once the core system was working, new feature ideas
kept appearing, and prioritisation became harder. In that sense, the planning challenge was less
about shortfall and more about how to limit expansion after early success.

The project’s framing also became more precise over time. At first it could easily have been
described too broadly as “AI part identification,” but later I refined it into a more defensible
claim: an evidence-backed shortlist assistant.

I also did not treat early success on the core goals as sufficient in itself.
If the system cannot identify items accurately enough, people simply will not use it. For that reason,
after the core goals were reached I focused as much as possible on improving model performance.
Accuracy remained the most important quality criterion even while new features were being added.


## Q2. Technical challenge and solution

### (a) Most significant technical challenge

The most significant technical challenge was reliable label recognition under realistic secondhand
conditions. I initially approached the problem through OCR, but general-purpose OCR is more suited
to document-like text and the labels in this project varied greatly in layout, quality, and damage.
It was not realistic to hand-code label-specific logic for every variant, so error handling became
very difficult.

Introducing a VLM solved many of these problems, but not all of them. Damaged labels, partially
erased characters, and unclear text still remained hard cases, and in those cases the system could
retrieve the wrong product. I therefore added other signals, such as recognising the product body
itself, but this also had limits: when two products looked almost identical and differed mainly in
label text, body-image features alone were not discriminative enough.

A further challenge was score fusion. Once multiple models were used together, performance depended
not only on each component but on how their outputs were combined. Weighted sums were difficult to
calibrate: if I gave too much weight to label evidence, cases with occluded or invisible labels could
be pushed toward the wrong product. Using many models at once also created latency constraints.

### (b) Approach taken

I addressed this by moving away from a pure OCR-based approach and toward a hybrid retrieval design
that combines OCR, VLM-based image understanding, body-image similarity, metadata, and text-based
signals. OCR remained important, but it was no longer treated as the only decisive source of truth.

Concretely, I built a pipeline that accepts user images and metadata, performs preprocessing, OCR,
embedding generation, retrieval across multiple Milvus collections, weighted fusion, and ranking. I
also ran repeated experiments and tuning passes to adapt the weighting and search flow to this
project rather than using a generic configuration. When direct identification was weak, I added
catalog retrieval and agent orchestration so that the system could use additional evidence.

To address speed limits, I also kept lighter paths available where possible rather than always using
the heaviest multimodal route. Overall, the strategy was to combine imperfect signals in a practical
way rather than assuming that a single model would solve the problem completely.

### (c) Evaluate effectiveness

This approach was clearly effective in practical terms. Introducing a VLM solved many cases that were
hard under OCR-only logic, and iterative experimentation helped me gradually tune the system for this
specific project. I consider that optimisation work one of the project’s real strengths.

At the same time, the limits are also clear. Damaged or erased labels remain difficult, and cases
where the body is nearly identical but only the label differs are still challenging. Hand-tuned
weighted fusion also has clear limits, and there is a real trade-off between accuracy and speed.

So I would describe the approach as a successful and realistic prototype rather than a complete
fundamental solution. If I revisited the system, I would not rely only on orchestration. I would also
strengthen domain-specific fine-tuning or specialist components for label recognition and pursue more
systematic fusion calibration.


## Q3. Background literature

### Reference 1: CLIP / image-text embedding literature

One of the most significant references for my project was the line of work on CLIP-style vision-language representation learning, especially papers showing that images and text can be embedded in a shared space for retrieval. The core contribution is that semantic cross-modal search can be performed without training a narrow classifier for each label set.

This was significant because it helped justify why the project should use retrieval rather than closed-set prediction. It influenced my decision to use multimodal embeddings and to think of user images and text queries as evidence that can be compared within a broader retrieval architecture. This is a high-quality and reliable reference because it is a foundational, widely cited research contribution that has strongly influenced later open-source systems.

### Reference 2: BGE-M3 / multilingual retrieval literature

A second important reference was the BGE-M3 work and related multilingual retrieval material. Its main contribution is support for strong multilingual and multi-function text retrieval, which is especially useful when OCR text, metadata, Korean terms, English terms, and shorthand identifiers all coexist.

This mattered to my project because secondhand part data is messy and multilingual. It influenced the decision to retain a robust text retrieval channel rather than treating everything as image-only similarity. I would also class this as a high-quality reference because it is technical, directly relevant, and tied to an influential retrieval model used in current practice.

## Q4. Future directions

### (a) Medium and long-term possibilities or impacts

In the medium term, this project could improve listing assistance on secondhand platforms by reducing the effort required to identify technical items and by increasing the consistency of listing metadata. In the long term, the same architecture could support equipment maintenance, spare-parts lookup, circular economy workflows, and industrial inventory discovery.

These claims are justified because the system is not tied only to one UI or one narrow dataset. Its core contribution is an orchestration pattern for multimodal evidence in open-world identification.

### (b) Ways future students or professionals could expand it

The most concrete expansions are:

- a full audited human-review workflow with accept/edit/reject states
- stronger evaluation automation for retrieval, OCR, latency, and usability
- region-focused OCR or object detection for hard cases
- richer catalog grounding and document retrieval
- active learning from confirmed user corrections

These are natural extensions because they build directly on the present architecture rather than replacing it.

### (c) Advice for future builders

My main advice is to define the problem very narrowly and clearly from the start. AI models are
improving so quickly that features which once seemed special can become standard very fast. Because
of that, the key is not simply to keep adding capabilities, but to define exactly what problem is
being solved and under what constraints.

In my case, an important constraint was that company information could not leave the organisation,
which made on-premise and local-model use central. Under that kind of constraint, simply
orchestrating many weaker models through weighted sums may not be the strongest long-term strategy.
A more competitive direction may be to make one model or one component extremely strong within a
narrow specialist scope.

For that reason, I would advise future builders to focus not only on making the agent pipeline more
complex, but also on identifying the project’s core value and strengthening domain-specific
fine-tuning or specialist components when appropriate.


## Q5. Presentation

### (a) What I would include in a 10-minute presentation

I would include the problem motivation, retrieval-first framing, architecture overview, one clear search flow, one UI demo, and one evaluation summary slide. I would also include one slide on limitations, because the project’s credibility depends on showing where the evidence ends.

I would leave out overly detailed code internals, exhaustive model comparisons, and unvalidated claims. In a 10-minute slot, clarity matters more than completeness.

### (b) Design the presentation

I would use a short slide deck with around 8 to 10 slides. The media would include architecture diagrams, screenshots, and one simplified score/evidence example. The flow would be: problem, requirements, system design, implementation, demo, evaluation, limitations, future work. I would speak in a product-and-engineering style rather than a purely academic style, because the project’s value is easiest to understand when tied to the user workflow.

The limitation of this presentation approach is that it simplifies some technical nuance. For example, it would not fully explain every collection, retrieval channel, or model trade-off. However, for a short talk this trade-off is acceptable.

---

# 3. CM3070 Past Exam 2024-03

## Q1. Project report

### (a) Purpose of the literature review

The literature review should not merely prove that reading was done. Its purpose is to position the project in relation to prior work, identify what is already known, expose important limitations or gaps, and justify the design choices taken later in the report.

In my project, the literature review justified several crucial decisions: using retrieval instead of closed-set classification, treating OCR as uncertain evidence, using multimodal retrieval in a fine-grained domain, and designing for user verification rather than automated certainty.

### (b) What material made it into the literature review and what did not

The literature review should prioritise material that can justify the project’s central design
choices over time. For this project, that means keeping conceptually durable work on multimodal
retrieval, OCR uncertainty, vector search, and human-in-the-loop assistance near the centre.

By contrast, some material becomes outdated very quickly. Version-specific model comparisons,
short-term benchmark results, and implementation-focused comparisons can be useful during
engineering, but they are weaker as core literature-review evidence because the AI landscape changes
so rapidly. In practice, after model upgrades I sometimes observed stronger results than in earlier
experiments, which reduced the long-term value of some early comparison material.

So the best approach is to foreground durable conceptual literature while using fast-moving model
comparison material more narrowly as engineering support.


### (c) What belongs in an appendix

Appendices are suited to supporting detail that would interrupt the main argument if placed in the main report. Examples include extended tables, extra screenshots, detailed test outputs, additional qualitative examples, long prompts, or supplementary implementation notes.

### (d) What appendices I would include

For this project, suitable appendices would include extra OCR failure examples, additional retrieval outputs, artifact-bundle contents, and detailed environment or test records. I would use appendices because they improve reproducibility and transparency while keeping the main chapters focused.

## Q2. Project design

### (i) Which template did you choose and why?

I chose the CM3020 template **“Orchestrating AI models to achieve a goal.”** This fit my project because the key contribution is not training one new model from scratch. Instead, the project combines OCR, multimodal retrieval, vector search, metadata processing, catalog search, and tool orchestration into one practical system.

### (ii) Summarise the technical challenge

The technical challenge was to identify fine-grained parts from imperfect secondhand photos in an open-world environment. Similar-looking parts often differ only in small textual markings, but those markings are hard to capture reliably.

### (iii) Summarise the main solution

My main solution was a hybrid, retrieval-first architecture that fuses image, OCR, text, metadata, and caption signals, stores them in searchable vector collections, and returns evidence-backed Top-K candidates for user review.

### (iv) Detailed description of the technical solution

At a high level, the user interacts through a web interface with search, indexing, chat, and catalog views. The web client sends requests to a FastAPI backend. The backend exposes routes for authentication, hybrid indexing/search, catalog PDF indexing/search, and agent chat. The API layer delegates to the model package, where the hybrid-search orchestrator handles preprocessing, OCR, embedding creation, candidate retrieval, fusion, and ranking.

Milvus stores multimodal vectors and metadata across several collections rather than one monolithic table. This supports image retrieval, text retrieval, caption retrieval, and structured metadata-aware lookup. The catalog path indexes PDF-derived chunks for internal document search. The agent path can orchestrate tools and return evidence sources.

The overall design is intentionally modular. The frontend presents the user-facing workflow, the API provides stable interfaces, and the model layer can evolve internally. This structure also supports later experimentation, because retrieval components can be adjusted without rewriting the whole user experience.

## Q3. Evaluation

### (a) Reflect on evaluation

I evaluated the project through a combination of prototype testing, retrieval-oriented analysis, implementation validation, and critical reflection. These were appropriate choices because the project is not only an algorithmic experiment but also a working system. The methods used included regression tests for recent API/model changes, retrieval observations, latency instrumentation, and structured discussion of what has and has not yet been measured.

The results show that the system clearly functions as an end-to-end prototype and that retrieval-first assistance is a credible approach. However, the evaluation also shows that some important quantitative areas remain incomplete. If I were to improve the evaluation, I would add earlier automation for CER, latency percentiles, and controlled hybrid ablation.

I would present these results using tables that separate supported claims, partial evidence, and future work. This presentation approach is justified because it makes the evidence boundary explicit rather than hiding uncertainty.

### (b) Reflect on project materials management

I maintained the project through a structured repository, separating web, API, model, docs, and submission artifacts. I used version control to manage changes and preserve a traceable development history. I also kept report evidence and working documentation in separate folders so that the submission baseline and current working state would not be confused.

If I repeated the project, I would improve artifact automation further, especially for evaluation outputs. The main lesson is that code versioning alone is not enough; evaluation artifacts also need deliberate organisation.

## Q4. Self-reflection

### (i) What I achieved

I achieved the core planned goals relatively quickly and was then able to extend the project further
with the agent path and the catalog path. In other words, I did not just build a search experiment;
I built an end-to-end prototype across web, API, and model layers.

### (ii) Best achievement

My best achievement was the amount of experimentation and optimisation I put into the project.
I tried multiple models, compared them in practice, and kept tuning the system to fit the actual
problem rather than leaving it at a generic configuration. I would most strongly praise the way I
used experimental results to improve the system incrementally.

### (iii) Lowest-quality part and how to improve it

The weakest part of the project was that I relied heavily on orchestration while the core fine-tuning
or domain-specific specialist modelling remained comparatively weak. Combining multiple models helped,
but I think the deeper competitive value of a future version would come from making the core model
much stronger on the exact task.

If I had another chance, I would improve that part directly. Rather than only refining weighted-sum
orchestration, I would invest more in domain-specific fine-tuning, specialist label-recognition
components, and more systematic score calibration.


## Q5. Further work

### (i) Would I advise a rewrite?

I would not advise a full rewrite. The existing modular separation between frontend, API, and model package is a strong foundation. However, I would recommend targeted refactoring where evaluation automation and reviewed writeback workflows need cleaner interfaces.

### (ii) Two important future additions

Two important additions would be:

- audited human-review workflow with accept/edit/reject controls
- controlled multimodal benchmarking and region-focused retrieval experiments

These are important because they strengthen both product usability and scientific validation.

### (iii) Advice on how to approach the work

I would advise a future student to preserve the current architecture, define measurable success criteria before coding, and avoid introducing extra model complexity without a clear evaluation plan. They should also keep careful control over data versions, collection schemas, and artifact generation.

---

# 4. CM3070 Past Exam 2023-09

## Q1. Project Approach

### (a) Which template did you choose, and why?

I chose the CM3020 template **“Orchestrating AI models to achieve a goal.”** This was the right choice because my project is fundamentally about coordinating multiple AI components to solve a practical identification problem. The work is strongest at the orchestration level rather than at the level of inventing a single new model.

### (b) What route did you take to deliver the requirements?

I took a retrieval-first route. Instead of training a fixed classifier, I built a hybrid search pipeline that can combine image evidence, OCR text, captions, metadata, and vector search. The route also included a frontend for user interaction, a backend API, model-side orchestration, and later catalog and agent extensions.

### (c) Suggest one other approach you could have taken

An alternative approach would have been a supervised closed-set classification system trained on a curated dataset of known parts.

### (d) Compare and contrast your own solution with the alternative

I chose the retrieval-first route because it fits an open-world domain better. Secondhand platforms constantly encounter new or rare items, so a closed classifier would become brittle and expensive to maintain.

The advantage of my route is flexibility, evidence exposure, and better fit to uncertain real-world data. The disadvantage is that it is architecturally more complex and demands careful ranking and evaluation. A closed-set classifier might be simpler to explain and benchmark in a narrow dataset, but it would be less realistic for the actual problem.

## Q2. Evaluation and Testing

### (a) Describe the testing and evaluation you did

I used a mixture of engineering validation and retrieval-oriented evaluation. Engineering validation included regression tests for API and model behaviour, especially around safety and testability changes. I also used implementation-level instrumentation to capture stage timings in the hybrid search path. Retrieval evaluation included qualitative and baseline evidence showing that image-first assistance is useful but incomplete on its own.

### (b) Other approaches you could have taken

I could also have used a larger fixed benchmark with explicit Top-K accuracy, more systematic OCR CER measurement, broader latency analysis, and a user study around trust and effort reduction. These approaches may have produced stronger quantitative evidence. I did not complete all of them mainly because of time and integration complexity, but in hindsight they should have been operationalised earlier.

## Q3. Self Reflection

### (a) Describe briefly what you achieved

I produced a working prototype for secondhand part identification assistance, covering search, indexing, evidence-backed retrieval, catalog search, and agent-style orchestration.

### (b) Two best parts of the work

First, the architectural framing was strong: the system treats the problem as open-world retrieval rather than closed-set recognition. Second, the implementation achieved meaningful end-to-end integration across multiple layers and modalities.

### (c) What more could you have done in another aspect?

I could have done more on evaluation automation and user-facing review workflow. Those areas are the clearest opportunities for improvement because they affect both scientific confidence and product maturity.

## Q4. Video

### (a) Primary intention of the video

The primary intention of the video should be to demonstrate that the project actually works and to make the architecture legible through concrete behaviour. The video is most valuable when it shows the system in action, not when it repeats the report.

### (b) Describe the video briefly

I would use a voice-over screen recording that shows the web interface, image upload, returned shortlist, evidence-backed results, and one example of agent or catalog use. This is the best choice because it keeps focus on the running prototype and allows the marker to see both usability and technical achievement.

### (c) Was five minutes short or long?

Five minutes is short if I try to show every component, but it is enough for a well-structured demonstration. The key is to select only the most important flows: one search scenario, one evidence example, one architecture summary, and one short note on limitations. I would leave out minor implementation detail so that the video remains clear and persuasive.

## Q5. Development

### (a) Two things I already knew how to do

Two abilities I already had were Python-based backend development and general machine-learning/data workflow thinking. These helped because the project required API integration, data handling, model orchestration, and technical debugging. Without these foundations, much more time would have been spent on basic setup rather than project-specific design.

### (b) Two things I had to learn during the project

I had to learn more about multimodal retrieval system design and about how to structure a project where evaluation evidence matters as much as implementation. I also had to improve my understanding of how OCR uncertainty interacts with retrieval ranking in a real application. I learned these through reading papers and technical documentation, experimenting in code, and iterating based on project feedback.

---

## Final revision reminders

When adapting these answers in a real exam, keep the following phrasing consistent:

- Describe the system as a **retrieval-first, human-in-the-loop identification assistant**.
- Do not claim that full quantitative evaluation is complete.
- Separate **implemented**, **instrumented**, and **future work** clearly.
- Emphasise that the contribution is mainly **system orchestration and practical integration**.
- Link technical decisions back to the domain: open-world inventory, fine-grained ambiguity, OCR uncertainty, and user need for verifiable results.
