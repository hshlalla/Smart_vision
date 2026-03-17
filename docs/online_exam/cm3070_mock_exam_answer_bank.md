# CM3070 Mock Exam Answer Bank

This document is a project-specific answer draft for the CM3070 mock exam, written for the
**Smart Image Part Identifier for Secondhand Platforms** project.
It is designed to complement `docs/online_exam/cm3070_past_exam_answer_bank.md`, but is organised
around the exact questions that appear in `submission/pastexam/CM3070 mock exam.pdf`.

## 0. Quick project identity

- **Project title:** Smart Image Part Identifier for Secondhand Platforms
- **Template used:** `CM3020 Artificial Intelligence, "Orchestrating AI models to achieve a goal"`
- **One-sentence framing:** a retrieval-first, human-in-the-loop assistant for identifying
  industrial and electronic parts from secondhand photos using OCR, multimodal retrieval,
  vector search, catalog evidence, and agent orchestration

## Reusable framing for this mock exam

- The project should be described as a **system integration project**, not as a novel foundation-model project.
- The most defensible claim is **evidence-backed shortlist assistance**, not fully autonomous identification.
- The strongest completed outcomes are:
  - end-to-end prototype across web, API, and model layers
  - OCR + image/text retrieval + Milvus-based search + catalog grounding
  - regression tests and latency instrumentation
  - safer indexing flow with metadata preview and duplicate-review merge decisions
- The main incomplete areas that should be described honestly are:
  - full aggregate OCR CER/WER reporting
  - full p50/p90/p95 latency summaries
  - large controlled hybrid ablation
  - full audited accept/edit/reject workflow
  - broader usability validation

---

# 1. Question 1 — Your project

## (a) What was the title of your project?

The title of my project was **Smart Image Part Identifier for Secondhand Platforms**.

## (b) Which template was your project based on?

My project was based on the **CM3020 Artificial Intelligence** template,
**“Orchestrating AI models to achieve a goal.”**

## (c) Briefly describe what your project was about.

My project addressed the problem of identifying industrial or electronic parts from secondhand
listing photos. This is a difficult problem because many parts look visually similar, while the
most important identifying cues often appear as small printed text such as model codes or part
numbers.

Instead of treating the problem as closed-set classification, I framed it as a **retrieval-first,
human-in-the-loop decision-support problem**. The system accepts user images and optional text,
extracts OCR evidence where possible, performs image and text retrieval, searches vector indexes,
and returns a shortlist of plausible candidates with supporting evidence. A web interface, API
layer, and model pipeline were built to support this workflow.

In the final implementation, this workflow also became more careful about repeated registrations.
If a newly uploaded part appears to match an already indexed part, the system can ask whether the
new upload should enrich the existing model rather than silently creating a fragmented duplicate.

---

# 2. Question 2 — Project process

## (a) Relationship to the template, flexibility, and constraints

My project related strongly to the chosen template because its main contribution was not training a
single new AI model, but **orchestrating several AI and search components into a useful end-to-end
system**. The template was therefore a very good fit. The project combined OCR, multimodal
retrieval, vector search, catalog retrieval, metadata-aware ranking, and an agent-style interaction
layer in order to help users identify secondhand parts.

The template had a meaningful level of flexibility. It gave a broad direction—use AI components in
combination to solve a practical problem—but it did not force one fixed architecture, dataset,
or model choice. That flexibility allowed me to choose a problem that was realistic and
technically interesting, and to adapt the architecture as the project evolved. For example, I could
move from a simple image-retrieval idea toward a more defensible hybrid retrieval system once it
became clear that OCR evidence and textual identifiers were critical in this domain.

At the same time, the template also imposed useful constraints. Because it was an orchestration
project, I needed to show a coherent workflow, justify why each component existed, and evaluate the
system as a whole rather than presenting a disconnected collection of tools. This constraint helped
keep the work grounded. It pushed me to think about the system’s practical output—namely a Top-K
shortlist with evidence—rather than trying to claim that the project solved identification in a
fully automated way.

The main advantage of flexibility was that it let me adapt the project to the real complexity of the
domain. I was able to integrate image retrieval, OCR, metadata processing, and catalog evidence in a
way that matched actual secondhand listing conditions. The main disadvantage was that flexibility
also made scope control harder. Because many extensions were possible, such as stronger OCR,
reranking, better review flows, more evaluation, and richer catalog grounding, there was a constant
risk of doing too much.

The constraints were valuable because they forced prioritisation. They made it more natural to build
an end-to-end prototype first and then add carefully chosen improvements. In practice, this meant I
ended up with a more honest and defensible project: not a complete production system, but a working
retrieval-first assistant with meaningful evidence and clear boundaries.

## (b) Inclusive design and radical inclusion

**Inclusive design** is the practice of designing systems so they can be used effectively by a wide
range of people, including users with different abilities, backgrounds, devices, levels of domain
knowledge, and situational constraints. In computing systems and apps, this means thinking beyond an
“average user” and accounting for variation in language, accessibility needs, technical confidence,
network quality, and context of use. In research, inclusive design also means choosing methods and
evaluation criteria that do not silently exclude certain user groups.

**Radical inclusion** goes further. It is not only about making a product usable by more people, but
about treating potentially excluded people and edge cases as central rather than peripheral. In
computing research, this means asking whose needs are usually ignored, whose data is underrepresented,
and whose failure cases are being hidden by a system that appears to work for an easier mainstream
case. Radical inclusion also implies being reflective about power and assumptions in the design
process itself.

Applied to computer systems, these ideas encourage designers to avoid brittle assumptions such as:
users always have perfect input data, users are technical experts, users speak one language, or users
will trust opaque outputs without evidence. Applied to research, they encourage more transparent
claims, broader consideration of failure cases, and stronger attention to real-world diversity.

## (c) Relationship of these ideas to my own project

To be honest, I did not begin this project by consciously adopting inclusive design or radical
inclusion as formal theoretical frameworks. The project started more from a practical workplace
perspective: while working in industry, I felt that a tool with these capabilities really should
exist. So the design motivation came more from solving real user pain points than from explicitly
applying inclusion theory.

Even so, some of my design choices ended up aligning with inclusive-design principles. The system
does not force one opaque answer. Instead, it returns a Top-K shortlist with evidence so that users
can inspect and verify the result themselves. It also assumes that inputs may be imperfect: labels
may be damaged, partially erased, occluded, or photographed poorly. In that sense, the system was
not designed only for ideal expert users with perfect data.

So the most honest answer is that I did not explicitly design the project around inclusive-design
terminology, but several practical decisions moved in that direction anyway. Showing evidence,
allowing user verification, and avoiding automatic commitment under uncertainty are all examples.

At the same time, I would not claim that the project fully implements radical inclusion. I did not
conduct a formal accessibility audit, and I did not complete a systematic usability study across a
broad range of user groups. In the exam, I would therefore describe the connection carefully: these
ideas were not the formal starting point of the project, but some of the resulting design choices
are compatible with them.


---

# 3. Question 3 — Aims, outcomes, and lessons learned

## (a) Comparison of final outcomes with initial aims and objectives

In my view, there was no major core feature that clearly failed against the original plan. In fact,
I reached the main planned goals faster than I expected, and after that I kept adding extensions
such as the agent path and the catalog path. So the most accurate description of the discrepancy is
not “I failed to meet the aims,” but rather “the core aims were met relatively quickly and the
project scope then kept expanding.”

The original goal was to build an AI-assisted system that could help identify industrial and
technical parts from secondhand images. That goal was achieved. I built an end-to-end prototype with
a web interface, an API layer, and a model/retrieval pipeline. The system can accept images and
optional text, perform retrieval, and return a shortlist with supporting evidence.

After reaching that core workflow, I extended the system further with agent orchestration, catalog
retrieval, regression testing, and latency instrumentation. This means the final outcome went beyond
what I first needed for a basic MVP.

At the same time, I kept focusing as much as possible on improving model performance.
My reason was simple: even if the core goals are achieved quickly, if the system cannot identify items
accurately enough, people will not actually use it. In other words, feature expansion alone was not
enough; usable accuracy remained the most important quality threshold for the project.

For that reason, the real difference between the initial aims and the final result is expansion
rather than shortfall. The harder project-management problem was prioritisation. Once additional
features became possible, it was difficult to decide which ideas should still fit within the time
available. So functionally the project met its main aims, but managerially the bigger challenge was
scope control.

## (b) Lessons learned: three aspects I would approach differently

### Aspect 1 — Problem definition and core value

If I did a similar project again, the first thing I would improve is the clarity of the problem
statement. AI models are improving so quickly that features which once looked novel can become
standard surprisingly fast. Because of that, it is not enough to keep adding capabilities. It is
more important to define exactly what problem is being solved, for whom, and what the core value of
the system really is.

Next time, I would lock down that problem definition earlier. That would make later decisions about
agents, catalogs, OCR, and reranking more coherent, because every feature would be judged against a
clear central purpose.

### Aspect 2 — Core model strategy and domain specialisation

The second thing I would change is how early I think about strengthening the core model rather than
mainly relying on orchestration. In this project, I worked under an on-premise constraint because
company information could not be allowed to leave the organisation. That pushed me toward local
models, which often performed worse than GPT-class hosted models.

My response was to combine multiple signals through orchestration and weighted fusion. This worked to
an extent, but it also created new problems: weight setting became difficult, one signal could hurt
another in failure cases, and latency increased as more models were added. If I repeated the
project, I would explore domain-specific fine-tuning or stronger specialist components earlier,
rather than assuming that combining many weaker components is always the best answer.

### Aspect 3 — Scope control and prioritisation

The third lesson is that I would manage scope more strictly. Because the initial goals were reached
faster than expected, new feature ideas kept appearing: the agent, the catalog path, extra UI flows,
and supporting features. These additions improved the project, but they also made it harder to keep
a clear boundary around what absolutely had to be finished within the submission timeline.

Next time, I would define milestone boundaries more strictly and delay any extension that does not
directly strengthen the core problem solution. That would help the project remain ambitious without
weakening its final completeness.


# 4. Question 4 — Background material and planning your project

## (a) Scope and usefulness of background reading

My feeling is not that some background reading was completely useless, but that certain types of
material lost value very quickly because AI was developing so fast. In particular, model-comparison
results, version-specific benchmarks, and implementation-oriented comparisons could become outdated
within a short time as newer model versions appeared. In practice, after upgrading the model stack, I
sometimes saw stronger results than in earlier experiments, which meant that some earlier comparison
material became less useful as a basis for final design decisions.

That does not mean the background reading was unimportant overall. More durable conceptual material —
for example on retrieval, OCR uncertainty, multimodal search, and human review — remained valuable.
The real lesson was that I should not treat rapidly changing engineering comparisons and more stable
conceptual literature as if they had the same long-term value.

If I repeated the project, I would keep the literature review more strongly centred on enduring
conceptual foundations, while using fast-moving model-comparison material more narrowly as
implementation support.

## (b) Planning time allocation at the start of the project

Time allocation was difficult at the start not simply because there were many tasks, but because it
was hard to separate the true core problem from the surrounding optional features. With OCR, VLM,
retrieval, and local-model constraints, the real effort of each component was difficult to estimate
before implementation and experimentation.

I therefore decided to allocate time first to the core question: could I solve the identification
problem under on-premise and local-model constraints? In practice, that meant prioritising the core
retrieval pipeline before adding further layers. I still think that was the right decision, because a
stable core workflow was necessary before features such as the agent path or the catalog path could
be justified.

However, once the initial plan started succeeding, the harder issue shifted from time allocation to
scope control. New adjacent ideas kept appearing, and prioritisation became more difficult than the
original scheduling problem. If I repeated the project, I would still focus on the core pipeline
first, but I would set much stricter limits on what extensions are allowed within the same project
cycle.
