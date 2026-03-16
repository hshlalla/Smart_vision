# CM3070 Additional Practice Answers

This file stores extra CM3070-style practice questions that are not directly taken from the current
past exam or mock exam PDFs, but are still useful for project-specific revision.

---

## Q2.1 Proposal vs final report

### Question

What is the main difference between the proposal for a project (whether as a video pitch or a written document)
and the final report (again, whether considering the written document or the video aspect of this)?
Using your own project, give examples to illustrate this difference. Also comment on how your proposal
related to your final report.

### Answer

The main difference is that a **proposal** is about what you **plan and expect** to do, whereas a
**final report** is about what you **actually built, evaluated, learned, and can now justify with
evidence**.

A proposal is therefore prospective. Its purpose is to define the problem, explain why the project is
worth doing, justify the chosen direction, and show that the work appears feasible within the
available time. It often includes intended aims, planned methods, expected architecture, and a rough
idea of evaluation. A proposal video pitch has the same role in a shorter format: it persuades the
listener that the idea is worthwhile and achievable.

By contrast, the final report is retrospective and evidential. It must describe the implemented
system, explain how the design changed during development, present the actual evaluation carried out,
and critically analyse both strengths and limitations. A final project video is also different from a
proposal pitch because it should show the system working, not merely describe a plan.

My own project shows this difference clearly. In a proposal version of my project, I would have said
that I intended to build an AI-assisted system for identifying industrial or electronic parts from
secondhand photos, likely using OCR, multimodal retrieval, vector search, and a shortlist-based user
workflow. At proposal stage, those elements would mainly be justified as a promising direction. The
focus would be on the problem, the motivation, the expected architecture, and why the project seemed
technically interesting and feasible.

In the final report, however, I can no longer speak only in terms of intention. I need to explain
what was actually built: a web interface, a FastAPI backend, a reusable model package, hybrid
retrieval across multiple signals, async indexing, catalog retrieval, and an agent-assisted path. I
also need to explain what changed in practice. For example, OCR was initially an important planned
signal, but during implementation it became clear that OCR alone was not reliable enough for damaged,
erased, or highly variable labels. That pushed the project toward a more hybrid approach using VLM
signals, body-image features, metadata, and evidence-backed ranking.

The same difference appears in the video aspect. A proposal video would mainly explain the concept,
expected workflow, and intended impact. A final video, by contrast, should show the project actually
working: for example, uploading an image, retrieving candidates, showing evidence-backed shortlist
results, and demonstrating the catalog or agent flow. In other words, the proposal video sells the
plan, whereas the final video demonstrates the achieved system.

In my project, the proposal and final report are related through the same core problem and the same
overall direction. Both are centred on secondhand part identification under the CM3020 theme of
orchestrating AI models to achieve a goal. The proposal would have introduced the retrieval-first and
human-in-the-loop idea; the final report keeps that core framing, but makes it much more precise,
because it is informed by real implementation, evaluation, and limitations.

I would therefore say that my final report grew out of the proposal, but did not remain identical to
it. The core motivation stayed stable, but the final report became broader, more concrete, and more
critical. It includes features that were added later, such as catalog retrieval and agent
orchestration, and it also explains what remained incomplete, such as full OCR aggregate benchmarking,
latency percentile summaries, and a complete audited review workflow. That is exactly why the final
report is stronger than the proposal: it is no longer a promise, but an evidence-based account of
what the project became.

---

## Q2.2 Ethics in design and development

### Question

The consideration of ethics is an important part of design and development. What are the most important ethical aspects to consider in software design and development? Which specific ethical considerations did you take into account in your project? Are there other ethical considerations you could or should have taken into account that you decided not to? Explain why.

### Answer

Ethics is important in software design and development because software does not only implement
technical logic; it also affects people, organisations, and decision-making. The most important
ethical aspects usually include **privacy and confidentiality**, **safety and reliability**,
**transparency and accountability**, **fairness and bias**, **security**, and the extent to which
humans remain meaningfully in control when a system is uncertain.

In my own project, the most important ethical issue was **confidentiality**. The project was shaped
by a practical constraint that company-related information should not leave the organisation. That is
one reason why on-premise and local-model deployment became important in the design. Ethically, this
matters because convenience alone would not justify exposing sensitive internal information to
external services.

A second important ethical consideration was **the harm caused by incorrect identification**. If the
system confidently returns the wrong part, it can mislead users, waste effort, and damage trust. For
that reason, I did not frame the project as a fully autonomous identifier. Instead, I designed it as
a retrieval-first, human-in-the-loop assistant that returns a shortlist with evidence. This is an
ethical as well as a technical choice, because it reduces the risk of presenting uncertain results as
if they were certain.

A third ethical consideration was **safe writeback and data integrity**. If uncertain model outputs
are automatically stored as trusted knowledge, the system can reinforce its own errors. In my
project, this was addressed by making writeback opt-in by default and by treating user confirmation
as important. This reflects the principle that uncertain outputs should not silently become permanent
system knowledge.

I also took **transparency** seriously. The system is designed to show evidence, sources, and ranked
candidates rather than only a single opaque answer. That matters ethically because users need a fair
chance to inspect and challenge the system’s output.

There are other ethical considerations that I could or should have addressed more fully. One is
**inclusive design and accessibility**. I did not begin the project from a formal accessibility
framework, and I did not complete a broad usability study across different user groups. Another is
**fairness and dataset bias**. Although I considered difficult cases and open-world variability, I
did not run a full ethical analysis of whether some item types, photo conditions, or user contexts
were systematically underserved. A third is **environmental cost and efficiency**, because multimodal
models and repeated experimentation can consume significant compute.

The reason I did not address all of these equally was mainly prioritisation under limited project
scope. I focused first on technical feasibility, confidentiality, and safe use under uncertainty,
because those were the most immediate constraints in my context. That does not mean the other ethical
issues are unimportant. Rather, they remain important areas for future improvement if the system were
to move closer to real deployment.

---

## Q2.3 Background reading and literature survey

### Question

Explain the importance of doing background reading and a literature survey in a computer science project. Describe the most challenging aspect you faced when writing your literature survey, and how you overcame this.

### Answer

Background reading and a literature survey are important because they place a computer science
project within existing knowledge instead of presenting it as an isolated implementation. They help
justify design choices, define suitable evaluation methods, and explain what gap the project is
trying to address. In my project, this was important for arguing that the task should be treated as
retrieval-first rather than as simple closed-set classification, and that OCR should be treated as
uncertain evidence rather than as a complete solution.

The most difficult part of writing the literature survey was that the AI field was changing very
quickly. Model versions, benchmark results, and engineering practices evolved so fast that some
material became less useful within a short time. It was also difficult to decide what belonged in
the literature survey and what was only an implementation detail.

I overcame this by focusing on durable conceptual themes, such as retrieval versus classification,
OCR uncertainty, multimodal retrieval, and human-in-the-loop design. I still used recent model
comparisons when helpful, but only as supporting context rather than as the core academic
foundation. That made the survey more stable and academically defensible even as the model landscape
continued to change.

---

## Q2.4 Ethical selection and reporting of background material

### Question

Discuss one ethical consideration in selecting and reporting background material, and explain your approach to addressing this consideration in your own project.

### Answer

One important ethical consideration in selecting and reporting background material is **honesty in
representation**. This means avoiding cherry-picking sources that only support the story I want to
tell, while ignoring evidence, limitations, or comparisons that make the project look less strong.
A literature survey should not exaggerate maturity, originality, or certainty.

In my project, I addressed this by separating durable conceptual literature from fast-moving model
comparisons. I used more stable sources to support the main framing, such as multimodal retrieval,
OCR uncertainty, and human-in-the-loop assistance. By contrast, short-term benchmark results and
model-version comparisons were treated more cautiously, because they can quickly become outdated.

I also tried not to overclaim what the background material proved. I did not use the literature to
suggest that the project had already solved fully autonomous identification. Instead, I used it to
support a more careful and defensible claim: that the system is best understood as a retrieval-first,
human-in-the-loop identification assistant.

---

## Q3.1 Evaluation case study

### Question

Consider the following case study:
Imagine that your project is the development of a learning analytics algorithm to detect students at risk of failure.
Your evaluation consists of asking reviewers to take a set of inputs to the algorithm and their corresponding outputs,
and answer the following sets of questions.

Set 1 consists of the following questions:
- Does the output help me understand which students are at risk?
- Would this output help a teacher support students better?
- Would this output help you improve your own student retention?

Set 2 consists of the following questions:
- Do you think student A is in need of additional support?
- Do you think student B is in need of additional support?
- Do you think student C is in need of additional support?

Critically discuss the tasks presented, and the questions asked, in the context of the evaluation approach that is being taken.
Include suggestions for additional or different approaches to the evaluation.

### Answer

This evaluation mainly measures reviewer perception rather than model validity.
Set 1 is useful for face validity and stakeholder acceptance, because it asks whether the output feels understandable and useful.
However, the questions are broad and subjective, and they do not separate explanation quality, interface design, and predictive accuracy.
Set 2 is closer to a decision task, because reviewers must judge whether individual students need support.
Even so, without ground truth, a rubric, or reviewer calibration, it still measures opinion more than correctness.
It may also reproduce reviewer bias and hides important issues such as uncertainty, false positives, and false negatives.
A stronger evaluation would combine qualitative review with quantitative evidence, such as precision, recall, calibration, and comparison with known outcomes.
It would also help to compare the system against teacher-only judgement, collect confidence ratings, measure inter-rater agreement, and test fairness across student groups.
A scenario-based study of whether teachers actually make better interventions after seeing the output would be especially valuable.

---

## Q3.2 20-minute in-person presentation

### Question

Imagine you are required to give an in-person presentation of 20 minutes to discuss the most significant aspects of your project.
Outline, with concrete detail, what this presentation would consist of, and which parts of the project you would highlight.
Justify your choices of what you would include. Also comment on what aspects would be most challenging to present.

### Answer

In a 20-minute presentation, I would spend about 2 minutes on the problem context and why accurate identification matters in practice.
I would then spend 3 minutes explaining why an OCR-only approach was not sufficient for damaged labels, erased text, and visually similar products.
The next 5 minutes would cover the core architecture: the web interface, FastAPI backend, reusable model package, and hybrid retrieval pipeline.
I would use another 4 minutes to explain the main technical challenge, which was combining OCR, VLM, body-image, and metadata signals.
Then I would spend 3 minutes on evaluation and representative failure cases, because they show both the progress made and the remaining limitations.
The final 3 minutes would cover the later extensions, such as catalog retrieval and the agent flow, plus the most important future work.
I would highlight these parts because they best show the problem definition, the engineering decisions, and the reasoning behind the final system.
The most challenging aspect to present would be the weighting and fusion trade-offs, because they are central to performance but easy to overcomplicate.

---

## Q3.3 Further work

### Question

Based on the significant aspects of your project identified in part (b) of this question, suggest further work, with justification,
that another student might do to continue or further advance the outcome in some way. Be as explicit as possible.

### Answer

One strong continuation would be to build a specialist module for damaged or partial labels, supported by a benchmark of hard cases.
This is justified because the main remaining weakness is not the lack of features, but weak discrimination in the exact failure cases that matter most.
A second continuation would be to replace the current weighted-sum fusion with a learned or calibrated fusion method based on signal quality and confidence.
This would matter because fixed weights are difficult to tune and can amplify the wrong evidence when one signal is unreliable.
A third continuation would be a stricter evaluation and feedback pipeline, including failure taxonomy, latency profiling, reviewer workflow, and safe writeback.
That would help move the system from a strong prototype toward a more reliable and deployable identification assistant.

---

## Q4.1 Computer science project vs IT deployment

### Question

Explain what makes a project a computer science project rather than an IT deployment. Discuss, with justification, where your project fits within this range and what implications this has for future work on the same project.

### Answer

A computer science project goes beyond installing or configuring existing technology.
It involves problem formulation, computational design choices, trade-off analysis, and evaluation of methods.
By contrast, an IT deployment mainly focuses on putting known tools into operation for a practical need.
My project sits between these two, but it is closer to computer science.
I did not only deploy OCR or retrieval tools; I reformulated the task as a retrieval-first, human-in-the-loop identification problem.
I also designed a hybrid pipeline that combines OCR, VLM, body-image, and metadata signals, and I evaluated its failure modes.
At the same time, it includes deployment aspects such as a web app, backend, and local operation.
This has two implications for future work: one path is research-oriented, such as better fusion and hard-case benchmarks, and the other is deployment-oriented, such as usability, monitoring, and operational robustness.
Because the project already has a strong method-design component, it can continue as a computer science project rather than only as an IT rollout.

---

## Q4.2 Radical inclusion vs born-accessible

### Question

In terms of inclusive design, explain the core similarities and differences between the radical inclusion approach and the born-accessible approach to inclusive software design. Include examples to illustrate the distinctions.

### Answer

Both radical inclusion and born-accessible design reject the idea that accessibility should be added only at the end.
Both aim to reduce exclusion from the beginning of the design process.
The main difference is emphasis.
Born-accessible focuses on making the product accessible from day one, often through features such as keyboard support, screen-reader labels, captions, and colour contrast.
Radical inclusion goes further by starting with people who are most likely to be excluded and letting those edge cases reshape the whole design.
For example, born-accessible in my project would mean an interface that is readable, navigable, and understandable from the start.
Radical inclusion would additionally ask how the workflow should support users facing damaged labels, uncertain outputs, or limited technical expertise, and may redesign the process around those difficult cases.
So born-accessible is accessibility-by-default, while radical inclusion is margin-first design that can change the product concept more deeply.

---

## Q4.3 One significant theoretical construct

### Question

Identify ONE significant theoretical construct used in your project and explain its role in the project. Discuss why the construct was significant, and whether it was the best or the only option you could have chosen. If it was the best or only option, describe why this was the case; and if it wasn't then discuss what other options were available and why you did not choose them.

### Answer

One significant theoretical construct in my project was the **human-in-the-loop decision-support model**.
Its role was to frame the system as an assistant that retrieves and ranks candidate parts with evidence, rather than as a fully autonomous classifier.
This was significant because my problem is open-world, labels may be damaged, and incorrect identification can mislead real users.
It was not the only possible option.
I could have chosen a closed-set classifier or a fully automated end-to-end model that always outputs one answer.
I did not choose those because they encourage overconfident single-label predictions and are brittle when labels are missing or products look very similar.
For my context, the human-in-the-loop construct was the best option because it matched the uncertainty of the task and still allowed practical use.

---

## Q4.4 One algorithm or method used

### Question

Describe one algorithm or method you used in the app, program, or system that you designed and developed, and explain your choice process, justifying the decisions you made about using that algorithm or method.

### Answer

One method I used was **weighted fusion ranking** across multiple signals.
OCR text similarity, VLM or image similarity, body-image features, and metadata each produced partial evidence, and I combined them into a ranked candidate list.
I chose this method because it was transparent, easy to inspect, and workable with limited labelled data.
It also let me add or remove signals without redesigning the whole system.
I considered more complex learned fusion or end-to-end multimodal models, but they required more data, more tuning, and were harder to debug.
Weighted fusion was therefore a practical and explainable choice for an uncertain, open-world problem, even though tuning the weights was difficult.

---

## Q4.5 Reproducibility

### Question

Reproducibility refers to the extent to which an algorithm, method, or tool can produce the same result when used again under the same, or similar, conditions. Discuss how this concept applies to the algorithm or method you described in Part(d) above.

### Answer

Reproducibility applies directly to weighted fusion ranking because, in principle, the same inputs, preprocessing steps, model versions, and weights should produce the same ranked output.
However, reproducibility can be weakened if model versions change, indexes are updated, OCR behaviour varies, or some model components are nondeterministic.
This means reproducibility depends not only on the formula but also on controlling the wider pipeline.
In my project, good reproducibility would require fixed benchmark data, logged weights, saved model versions, stable configuration, and documented preprocessing.
The method is more reproducible than informal human judgement, but less reproducible than a fully deterministic rule system if the underlying models keep changing.
So reproducibility is achievable, but only when the full experimental context is carefully fixed and recorded.
