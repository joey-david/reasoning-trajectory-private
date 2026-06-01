# Zotero Notes

## Making Slow Thinking Faster: Compressing LLM Chain-of-Thought via Step Entropy

- Type: `conferencePaper`
- Key: `U37M28DQ`
- Creators: Zeju Li, Jianyuan Zhong, Ziyang Zheng, Xiangyu Wen, Zhijian Xu, Yingying Cheng, Fan Zhang, Qiang Xu
- URL: https://openreview.net/forum?id=cGLqQfS5wH

## Training Large Language Model to Reason in a Continuous Latent Space

- Type: `journalArticle`
- Key: `3PWNTJ98`
- Creators: Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason E. Weston, Yuandong Tian
- URL: https://openreview.net/forum?id=tG4SgayTtk

### Note: CoCoNUT: Doesn’t work? See (Zhang et al., 2025).

CoCoNUT: Doesn’t work? See (Zhang et al., 2025).

## solution object rate distortion limits - Online LaTeX Editor Overleaf

- Type: `webpage`
- Key: `VTWY9CA8`
- URL: https://www.overleaf.com/project/6a03318248b03f2bd7094bb2/detacher

## ChatGPT

- Type: `webpage`
- Key: `YIKJEQ3L`
- URL: https://chatgpt.com/

## solution object rate distortion limits - Online LaTeX Editor Overleaf

- Type: `webpage`
- Key: `ER5PZNNM`
- URL: https://www.overleaf.com/project/6a03318248b03f2bd7094bb2/detacher

## Chain Of Thought Compression: A Theoritical Analysis

- Type: `preprint`
- Key: `EZ46KWBE`
- Creators: Juncai Li, Ru Li, Yuxiang Zhou, Boxiang Ma, Jeff Z. Pan
- URL: http://arxiv.org/abs/2601.21576

## Emergent Analogical Reasoning in Transformers

- Type: `webpage`
- Key: `M3AR86B9`
- Creators: Gouki Minegishi, Jingyuan Feng, Hiroki Furuta, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo
- URL: https://arxiv.org/abs/2602.01992v4

### Note: $\textbf{*1}$: Categories are abstract mathematical structures made of objects and morphisms between objects. 

$\textbf{*1}$: Categories are abstract mathematical structures made of objects and morphisms between objects. 

A category:- Objects A, B, C, …- $\forall$ pair $\{A, B\}$ of objects, a collection of morphisms $f: A \rightarrow B$- a composition operation that ensures associativity and the existence of an identity morphism $\forall$ objects.

A functor (e.g. $F$) is a structure-preserving map between categories (NOT between objects). Sends objects in a category to other objects in a category, same with arrows/morphisms, preserves the identity mapping and composition, e.g. $F(g \circ f) = F(g) \circ F(f)$.

$\textbf{*2}$: is not about CoT reasoning or even reasoning path, but about emergent reasoning in the structure of the embedding space. Says that analogies are enabled by functors between similar structures in hidden space. See if still useful (yes, at least as an interp paper).

$\textbf{*3}$: There’s a mathematical idea to be pulled out here, on the property of generalization as an increasing connectedness between more and more distant neurons/layers, allowing to see more and more abstract patterns as training evolves - e.g. the model secures the easy wins early on (memorization, n-gram like learning), then easily neurally-reconstructable structures like compositional reasoning, and ends with complex structures between embedded objects, like analogy. Will need to look up what’s already been done on this.

$\textbf{*4}$: but the Dirichlet energy of $\textit{what}$?

$\textbf{*5}$: Clearly drawing analogy between compositional reasoning and CoT reasoning as a pathway of hops between atomic facts/latent positions. 

$\textbf{*6}$: Notice the log scale for training steps + on their synth dataset, analogical reasoning only really seems to start after training and compositional reasoning have saturated accuracy.

## Understanding and Steering the Cognitive Behaviors

- Type: `document`
- Key: `8P4BIDYN`

## awq

- Type: `document`
- Key: `X4PCOP1E`

## fm lora asymetry

- Type: `document`
- Key: `G6DNYLIA`

## gptq

- Type: `document`
- Key: `UDYCM6CR`

## intrinsic dim eff finetuning

- Type: `document`
- Key: `H0964BBC`

## lbllm survey

- Type: `document`
- Key: `VBD6VCXL`

## mamba ssm

- Type: `document`
- Key: `G8Q8H4MH`

## mdl revisited

- Type: `document`
- Key: `0F8GEJCR`

## pac bayes comp bounds

- Type: `document`
- Key: `WWRR0LFX`

## paretoQ

- Type: `document`
- Key: `UCC10ZTT`

## radio optim

- Type: `document`
- Key: `WC7VRVBU`

## smooth quant

- Type: `document`
- Key: `1O6VMT4H`

## strong gen bounds deep nets

- Type: `document`
- Key: `WI7ADW3A`

## transformers succint

- Type: `document`
- Key: `ZUOGZJ9W`

## zeroquant

- Type: `document`
- Key: `PCA9BWDI`

## CoT Harms Performance of Rather Smaller Language Models

- Type: `conferencePaper`
- Key: `64W3X3AS`
- Creators: Jihoo Shim, Shin Dong Ho, Jeongwon Kim
- URL: https://index.ieomsociety.org/index.cfm/article/view/ID/25358

## Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

- Type: `webpage`
- Key: `YA363CGR`
- Creators: Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou
- URL: https://arxiv.org/abs/2201.11903v6

## ReAct: Synergizing Reasoning and Acting in Language Models

- Type: `webpage`
- Key: `ZSINCCAS`
- Creators: Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
- URL: https://arxiv.org/abs/2210.03629v3

### Note: *1: How does the trajectory get affected/how does it jump when prefill with the tool feedback is added?

*1: How does the trajectory get affected/how does it jump when prefill with the tool feedback is added?

*2: CoT is a static black box in that it never benefits from new information from outside of the model, relying solely on its prebuilt representations. ReAct, solves this by querying outside information from functions - I think of this as using discrete “pulls” of information from the real world to force-align an activation with a potential ground truth via prefill. Could there be another way to ensure stability and righteousness of direction without tool calls, one either already contained within the LLM, or a mathematical tool that could be used on its activations/weights? You’d assume training already optimizes the inner representations and pathways of the LLM as far as can be, but…

*3: Nowadays, all SoTA LLMs, especially those that solve erdos proofs or implement entire oses autonomously, run with integrated tools and verifiers. Using those surely grounds reasoning in the way described above, so how does it guide reasoning in latent space vs vanilla, hallucination-prone CoT? Think future-wise, LLMs are changing to be hybrid systems. Don’t be a purist for no reason, good conf paper can come from this.

## Explainable Chain-of-Thought Reasoning: An Empirical Analysis on State-Aware Reasoning Dynamics

- Type: `preprint`
- Key: `UPVMQ8GC`
- Creators: Sheldon Yu, Yuxin Xiong, Junda Wu, Xintong Li, Tong Yu, Xiang Chen, Ritwik Sinha, Jingbo Shang, Julian McAuley
- URL: http://arxiv.org/abs/2509.00190

### Note: Comment: 5 pages, 4 figures

Comment: 5 pages, 4 figures

## Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought

- Type: `preprint`
- Key: `3HUMLP66`
- Creators: Yuyi Zhang, Boyu Tang, Tianjie Ju, Sufeng Duan, Gongshen Liu
- URL: http://arxiv.org/abs/2512.21711

### Note: Comment: 13 pages, 5 figures

Comment: 13 pages, 5 figures

### Note: Seems to rely a lot on steering. Induces the ideas that on top of CoT for reasoning, I’ll need to look at representation

Seems to rely a lot on steering. Induces the ideas that on top of CoT for reasoning, I’ll need to look at representation + steering work (especially token-to-token evolution representation dynamics), as well as interpretability.

*1: disturb (add noise/variations) key tokens in reasoning and see how the trajectory degrades. Identify key tokens? In their case, they use it to show that latent tokens have no effect, but we could use it better.

See Liu et al 2024 for graph-based reasoning with CoT. May be be useful for proofs and program generation.

See Bråtelund, 2024 for early exploration of latent reasoning paths and hidden space intervention.

*2: how do you actually apply orthogonal perturbations? Look at the code, at the magnitude, etc.

## LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals

- Type: `preprint`
- Key: `EJHKPQ3U`
- Creators: Lihao Sun, Hang Dong, Bo Qiao, Qingwei Lin, Dongmei Zhang, Saravan Rajmohan
- URL: http://arxiv.org/abs/2604.05655

### Note: Comment: ACL 2026 (Main)

Comment: ACL 2026 (Main)

### Note: Obviously, this is basically exactly the idea I was having. Maybe it falls short in some places? Having just read the ab

Obviously, this is basically exactly the idea I was having. Maybe it falls short in some places? Having just read the abstract: they focus on predicting

1* Step-specific regions more and more separable with reasoning depth.

Latent solution paths/objects in reasoning LLMs: we study and find ways to compress/extract the reasoning trajectories that reasoning LLMs take when solving mathematical proofs, programs or other verifiable objects - we're not just interested in the solution, but in the solution process as well - there are multiple valid proofs to a theorem in math, but some are shorter/better than others, same with programs, some are more/less complex than others, etc. We aim to find out precisely how we could exploit this new view of CoT hidden trajectories, and see how well they map to solution objects (Yang et al.; Minegishi et al., 2026)

> reasoning correction and length control based on derived ideal trajectories



> lthough we observe clear and consistent trajectory structure in GSM8K, MATH-500, and MMLU, it remains an open question whether similar geometric organization arises in other settings, such as open-ended reasoning, multi-hop QA, or program synthesis.



> Finally, our trajectory-based interventions rely on estimating an ideal trajectory from correct training examples. While this assumption is empirically supported in our settings, it may break down when correctness is underspecified. Extending this approach to such settings may require richer notions of ideal behavior, or task-conditioned reference trajectories



## What do Language Models Learn and When? The Implicit Curriculum Hypothesis

- Type: `preprint`
- Key: `Z2LTD6FV`
- Creators: Emmy Liu, Kaiser Sun, Millicent Li, Isabelle Lee, Lindia Tjuatja, Jen-tse Huang, Graham Neubig
- URL: http://arxiv.org/abs/2604.08510

## How to think step-by-step: A mechanistic understanding of chain-of-thought reasoning

- Type: `preprint`
- Key: `PBA8JKVQ`
- Creators: Subhabrata Dutta, Joykirat Singh, Soumen Chakrabarti, Tanmoy Chakraborty
- URL: http://arxiv.org/abs/2402.18312

## Proximal Curriculum for Reinforcement Learning Agents

- Type: `preprint`
- Key: `U9MEHW57`
- Creators: Georgios Tzannetos, Bárbara Gomes Ribeiro, Parameswaran Kamalaruban, Adish Singla
- URL: http://arxiv.org/abs/2304.12877

### Note: Comment: Published in Transactions on Machine Learning Research (TMLR) 2023

Comment: Published in Transactions on Machine Learning Research (TMLR) 2023

## Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning

- Type: `preprint`
- Key: `KY5YQJ2I`
- Creators: Benhao Huang, Zhengyang Geng, Zico Kolter
- URL: http://arxiv.org/abs/2605.21488

### Note: Comment: ICML 2026

Comment: ICML 2026

## Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models

- Type: `journalArticle`
- Key: `PPRP46QB`
- Creators: Yukang Yang, Declan Campbell, Kaixuan Huang, Mengdi Wang, Jonathan Cohen, Taylor Webb

## Learn from your own latents and not from tokens: A sample-complexity theory

- Type: `preprint`
- Key: `PR8827XK`
- Creators: Daniel J. Korchinski, Alessandro Favero, Matthieu Wyart
- URL: https://arxiv.org/abs/2605.27734

