# Doubly Perturbed Task Free Continual Learning (AAAI 2024, oral)

### ✏️ [Project Page](https://hyun1a.github.io/dpcl.io) | 📄 [Paper](https://arxiv.org/abs/2312.13027)

> **Doubly Perturbed Task Free Continual Learning**<br>
> Byung Hyun Lee, Min-hwan Oh, Se Young Chun <br>
> 
>**Abstract**: Task Free online continual learning (TF-CL) is a challenging problem where the model incrementally learns tasks without explicit task information. Although training with entire data from the past, present as well as future is considered as the gold standard, naive approaches in TF-CL with the current samples may be conflicted with learning with samples in the future, leading to catastrophic forgetting and poor plasticity. Thus, a proactive consideration of an unseen future sample in TF-CL becomes imperative. Motivated by this intuition, we propose a novel TF-CL framework considering future samples and show that injecting adversarial perturbations on both input data and decision-making is effective. Then, we propose a novel method named Doubly Perturbed Continual Learning (DPCL) to efficiently implement these input and decision-making perturbations. Specifically, for input perturbation, we propose an approximate perturbation method that injects noise into the input data as well as the feature vector and then interpolates the two perturbed samples. For decision-making process perturbation, we devise multiple stochastic classifiers. We also investigate a memory management scheme and learning rate scheduling reflecting our proposed double perturbations. We demonstrate that our proposed method outperforms the state-of-the-art baseline methods by large margins on various TF-CL benchmarks.
<br>


## Getting Started
### Experiment environment
**OS**: Ubuntu 20.04 LTS

**GPU**: Geforce RTX 3090 with CUDA 11.1

**Python**: 3.8.15

To set up the python environment for running the code, we provide requirements.txt that can be installed using the command
<pre>
pip install -r requirements.txt
</pre>


### Experiments

**Disjoint continual learning**
<pre>
sh ./scripts/exp_script_blurry.sh
</pre>

**Blurry continual learning**
<pre>
sh ./scripts/exp_script_disjoint.sh
</pre>

**i-Blurry continual learning**
<pre>
sh ./scripts/exp_script_iblurry.sh
</pre>
