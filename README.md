Neural networks (NNs) often face a challenge called catastrophic forgetting, where performance on previously learned tasks degrades when new tasks are introduced without revisiting prior data. Continual learning (CL) addresses this issue by enabling models to learn sequentially from a stream of tasks while retaining prior knowledge. CL aims to balance the stability-plasticity dilemma, where stability ensures knowledge retention, and plasticity allows the integration of new information.

We evaluated 13 state-of-the-art CL methods including EWC, SI, LwF, BiC, PNN, DER++, iCaRL, GEM, PODNet, DyTox, RMM, Coil, and HAL—spanning the developmental timeline of CL approaches. For benchmarking, we used four datasets under different scenarios:

1. **CIFAR100: Tested under:
   - Task-Incremental Learning (TIL): 20 tasks (5 classes each) and 10 tasks (10 classes each).
   - Class-Incremental Learning (CIL): 5 classes per task and 10 classes per task.
   
2. **ImageNet1000*: Evaluated similarly to CIFAR100 with:
   - TIL: 20 tasks (50 classes each) and 10 tasks (100 classes each).
   - CIL: 50 classes per task and 100 classes per task.

3. **Permuted MNIST (P-MNIST): Creates 20 tasks with data transformed by unique pixel permutations, representing Domain-Incremental Learning (DIL).

4. **Rotated MNIST (R-MNIST): Consists of 20 tasks where each rotates the digits by a fixed angle between 0 and 180 degrees, also used to test DIL scenarios.

For comparison, we included Joint training, which trains all tasks together as an upper bound, and Finetuning, which sequentially trains on new tasks without addressing CF, as a lower bound. 


Efficient Lifelong Learning with A-GEM (A-GEM, A-GEM-R - A-GEM with reservoir buffer): agem, agem_r.
AttriCLIP: A Non-Incremental Learner for Incremental Knowledge Learning (AttriCLIP): attriclip.
Bias Correction (BiC): bic.
Continual Contrastive Interpolation Consistency (CCIC) - Requires pip install kornia: ccic.
Continual Generative training for Incremental prompt-Learning (CGIL): cgil
Contrastive Language-Image Pre-Training (CLIP): clip (static method with no learning).
CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning (CODA-Prompt) - Requires pip install timm==0.9.8: coda-prompt.
Generating Instance-level Prompts for Rehearsal-free Continual Learning (DAP): dap.
Dark Experience for General Continual Learning: a Strong, Simple Baseline (DER & DER++): der and derpp.
DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning (DualPrompt) - Requires pip install timm==0.9.8: dualprompt.
Experience Replay (ER): er.
Experience Replay with Asymmetric Cross-Entropy (ER-ACE): er_ace.
May the Forgetting Be with You: Alternate Replay for Learning with Noisy Labels (AER & ABS): er_ace_aer_abs.
Rethinking Experience Replay: a Bag of Tricks for Continual Learning (ER-ACE with tricks): er_ace_tricks.
online Elastic Weight Consolidation (oEWC): ewc_on.
Function Distance Regularization (FDR): fdr.
Greedy Sampler and Dumb Learner (GDumb): gdumb.
Gradient Episodic Memory (GEM) - Unavailable on windows: gem.
Greedy gradient-based Sample Selection (GSS): gss.
Hindsight Anchor Learning (HAL): hal.
Incremental Classifier and Representation Learning (iCaRL): icarl.
Image-aware Decoder Enhanced à la Flamingo with Interleaved Cross-attentionS (IDEFICS): idefics (static method with no learning).
Joint training for the General Continual setting: joint_gcl (only for General Continual).
Learning to Prompt (L2P) - Requires pip install timm==0.9.8: l2p.
LiDER (on DER++, iCaRL, GDumb, and ER-ACE): derpp_lider, icarl_lider, gdumb_lider, er_ace_lider.
Large Language and Vision Assistant (LLAVA): llava (static method with no learning).
Learning a Unified Classifier Incrementally via Rebalancing (LUCIR): lucir.
Learning without Forgetting (LwF): lwf.
Learning without Shortcuts (LwS): lws.
Learning without Forgetting adapted for Multi-Class classification (LwF.MC): lwf_mc (from the iCaRL paper).
Meta-Experience Replay (MER): mer.
Mixture-of-Experts Adapters (MoE Adapters): moe_adapters.
Progressive Neural Networks (PNN): pnn.
Online Continual Learning on a Contaminated Data Stream with Blurry Task Boundaries (PuriDivER): puridiver.
Random Projections and Pre-trained Models for Continual Learning (RanPAC): ranpac.
Regular Polytope Classifier (RPC): rpc.
Synaptic Intelligence (SI): si.
SLCA: Slow Learner with Classifier Alignment for Continual Learning on a Pre-trained Model (SLCA) - Requires pip install timm==0.9.8: slca.
Slow Learner with Classifier Alignment (SLCA): slca.
Semantic Two-level Additive Residual Prompt (STAR-Prompt): starprompt. Also includes the first-stage only (first_stage_starprompt) and second-stage only (second_stage_starprompt) versions.
Transfer without Forgetting (TwF): twf.
eXtended-DER (X-DER): xder (full version), xder_ce (X-DER with CE), xder_rpc (X-DER with RPC).
