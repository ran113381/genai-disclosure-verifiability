# Identification gold-standard coding codebook (binary)

Binary human labeling of whether an MD&A passage is a genuine enterprise GenAI disclosure, used to validate the **dictionary identification stage** (Stage 1) of the measurement pipeline and LLM baselines.

- **Purpose**: Create a human-labeled gold standard for evaluating the dictionary-based GenAI disclosure measure and LLM baselines.
- **Primary label values**: Use 1, 0, or U only.
- **1 = GenAI-related disclosure**: The passage clearly refers to enterprise-level generative AI, large models, AIGC, ChatGPT-like tools, content/code/image/text generation, intelligent Q&A/writing, model platforms/products, or business deployment/exploration of GenAI.
- **0 = Not GenAI-related disclosure**: The passage refers only to ordinary digitalization, generic AI, automation, intelligent manufacturing, cloud computing, big data, IoT, robotics, strategic slogans, background noise, or an unrelated/false keyword match.
- **U = Unclear**: The passage is too vague to decide. Use U only when the text is genuinely ambiguous.
- **Coder confidence values**: High / Medium / Low.
- **Blind coding rule**: Coders should use Annotation_Blind only. Do not look at Sampling_Metadata before independent coding.
- **Disagreement resolution**: After coder1 and coder2 finish, unresolved disagreements should be adjudicated by discussion or a third coder. The adjudicated value becomes final_label.
- **Metric calculation later**: Dictionary precision, recall and F1 require the final_label merged with Sampling_Metadata.
- **Important boundary**: Label disclosure relevance, not true internal deployment scale. The passage can be coded 1 if it substantively discloses GenAI-related exploration, adoption, products, platforms, or applications.

Sampling: 300 dictionary-hit (random) + 150 dictionary non-hit near-miss (hard AI-context) + 150 dictionary non-hit random. Two independent coders, all disagreements adjudicated (478 agreed + 122 adjudicated).