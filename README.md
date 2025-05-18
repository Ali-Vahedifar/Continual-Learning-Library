
# **Research 1) Towards Lifelong Deep Learning: A Review of Continual Learning and Unlearning Methods**

Neural networks (NNs) often face a challenge called catastrophic forgetting, where performance on previously learned tasks degrades when new tasks are introduced without revisiting prior data. Continual learning (CL) addresses this issue by enabling models to learn sequentially from a stream of tasks while retaining prior knowledge. CL aims to balance the stability-plasticity dilemma, where stability ensures knowledge retention, and plasticity allows the integration of new information.

We evaluated 13 state-of-the-art CL methods, including EWC, SI, LwF, BiC, PNN, DER++, iCaRL, GEM, PODNet, DyTox, RMM, Coil, and HAL—spanning the developmental timeline of CL approaches. For benchmarking, we used four datasets under different scenarios:

1. CIFAR100: Tested under:
   - Task-Incremental Learning (TIL): 20 tasks (5 classes each) and 10 tasks (10 classes each).
   - Class-Incremental Learning (CIL): 5 classes per task and 10 classes per task.
   
2. ImageNet1000: Evaluated similarly to CIFAR100 with:
   - TIL: 20 tasks (50 classes each) and 10 tasks (100 classes each).
   - CIL: 50 classes per task and 100 classes per task.

3. Permuted MNIST (P-MNIST): Creates 20 tasks with data transformed by unique pixel permutations, representing Domain-Incremental Learning (DIL).

4. Rotated MNIST (R-MNIST): Consists of 20 tasks where each rotates the digits by a fixed angle between 0 and 180 degrees, also used to test DIL scenarios.

For comparison, we included Joint training, which trains all tasks together as an upper bound, and Finetuning, which sequentially trains on new tasks without addressing CF, as a lower bound. 

# **Research 2) No Forgetting Learning:**

We introduce No Forgetting Learning (NFL), a memory-free CL framework (No access to prior data). NFL employs knowledge distillation to mitigate CF by preserving prior knowledge through soft-target supervision. Specifically, the NFL uses approximately 14.75× less memory than state-of-the-art methods. Building upon this foundation, NFL+ integrates an under-complete autoencoder (AE). NFL+ promotes knowledge retention by preserving essential feature representations. NFL+ ensures **knowledge retention**, **bias minimization**, and **stepwise freezing** for incremental learning settings. 

To better evaluate CL models, we propose the Plasticity-Stability Ratio, a new metric that quantifies a model’s ability to adapt to new tasks (plasticity) while maintaining performance on previous ones (stability).

## Training Pipeline Illustration

This figure illustrates the **stepwise freezing** sequential training pipeline of the NFL (5-step) and NFL+ (6-step) frameworks:

<p align="center">
  <img width="752" alt="NFL/NFL+ Training Pipeline" src="https://github.com/user-attachments/assets/c3dad354-8a99-4606-ba1f-ce86c0070398">
  <br>
  <em>Figure 1: Stepwise training pipeline of NFL and NFL+ frameworks. Blue boxes indicate task-specific inputs, green boxes represent currently trained parameters, and gray boxes denote frozen parameters.</em>
</p>

### Pipeline Stages:
1. **Step 1**: Initial training on task T<sub>t</sub> with random initialization (θ<sub>s</sub><sup>r</sup>, θ<sub>t</sub><sup>r</sup>)
2. **Step 2**: New task T<sub>t+1</sub> added with frozen (θ<sub>s</sub>, θ<sub>t</sub>)
3. **Step 3**: Knowledge distillation updates θ<sub>s</sub><sup>u</sup> and θ<sub>t</sub><sup>u</sup> using frozen θ<sub>t+1</sub>
4. **Step 4**: Joint fine-tuning of θ<sub>s</sub><sup>u</sup> and θ<sub>t+1</sub>
5. **Step 5**: Final refinement with dual soft targets (H<sub>t</sub> and H̃<sub>t</sub>)
6. **Step 6** (NFL+ only): Autoencoder regularization with feature transformation Γ(X<sub>t+1</sub>)



## **Update Roadmap:**
The codebase is actively being developed. We are adding new models to the repository to broaden its utility, We are creating an openly accessible dashboard to display the performance of all models in their respective configurations (to ensure reproducibility) and in general settings (for comparisons). This feature may take some time due to computational resource constraints.

Current Models: 
1) **No Forgetting Learning (NFL) (Our method)**
2) Gradient Episodic Memory (GEM)
3) Incremental Classifier and Representation Learning (iCaRL)
4) Bias Correction (BiC)
5) Hindsight Anchor Learning (HAL)
6) Learning without Forgetting (LwF)
7) Synaptic Intelligence (SI)
8) Progressive Neural Networks (PNN)
9) Elastic Weight Consolidation (EWC)
10) Co-Transport for Class-Incremental Learning (Coil)
11) Pooled Outputs Distillation for Small-Tasks Incremental Learning (PODNet)
12) Memory-Efficient Expandable Model (MEMO)

## **Prerequisites**
torch\
torchvision\
tqdm\
numpy

## **Citing the library:**

@misc{vahedifar_2025_14631802,\
  author       = {Vahedifar, Mohammad Ali and Zhang, Qi and Iosifidis, Alexandros},\
  title        = {Towards Lifelong Deep Learning: A Review of Continual Learning and Unlearning Methods},\
  month        = jan,\
  year         = 2025,\
  publisher    = {Zenodo},\
  doi          = {10.5281/zenodo.14631802},\
  url          = {https://doi.org/10.5281/zenodo.14631802},\
}

@misc{vahedifar2025forgettinglearningmemoryfreecontinual,\
      title={No Forgetting Learning: Memory-free Continual Learning}, \
      author={Mohammad Ali Vahedifar and Qi Zhang},\
      year={2025},\
      eprint={2503.04638},\
      archivePrefix={arXiv},\
      primaryClass={cs.LG},\
      url={https://arxiv.org/abs/2503.04638}, 
}
