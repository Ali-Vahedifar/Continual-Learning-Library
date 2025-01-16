
# **Towards Lifelong Deep Learning: A Review of Continual Learning and Unlearning Methods**

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


## **Update Roadmap:**
The codebase is actively being developed. We are adding new models to the repository to broaden its utility, We are creating an openly accessible dashboard to display the performance of all models in their respective configurations (to ensure reproducibility) and in general settings (for comparisons). This feature may take some time due to computational resource constraints.


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
