## Disentangled Sticky Hierarchical Dirichlet Process Hidden Markov Model

This code implements the disentangled sticky HDP-HMM (DS-HDP-HMM) and two baseline models: sticky HDP-HMM (S-HDP-HMM) and HDP-HMM.

## Installation

The code is written in Python 3.6. In addition to standard scientific Python libraries (numpy, scipy, matplotlib), the code expects the munkres package. You can install the munkres package using: `conda install -c conda-forge munkres`

To download this code, run `git clone https://github.com/zhd96/ds-hdp-hmm.git`

## Examples

There are examples for each model and emission function in the examples folder. We are working on more modulated code.

You can run the example through `nohup python file-name.py command-line-parameters &` in general. See each example file for the specific command line parameters it uses. For examples, the files in examples/ds-hdp-hmm folder are examples for DS-HDP-HMM with the following different observations and samplers:

| File name | Description |
| --- | --- |
| run_full_bayesian_gibbs_gaussian | Gaussian observation, direct assignment sampler |
| run_full_bayesian_gibbs_multinomial | multinomial observation, direct assignment sampler |
| run_full_bayesian_gibbs_poisson | Poisson observation, direct assignment sampler |
| run_full_bayesian_approx_parallel_gibbs_poisson | Poisson observation, weak-limit sampler in parallel |
| run_full_bayesian_approx_parallel_gibbs_ar | auto-regressive observation, weak-limit sampler in parallel |

The files in examples/s-hdp-hmm and examples/hdp-hmm are defined for S-HDP-HMM and HDP-HMM respectively as in the examples/ds-hdp-hmm.

The file in examples/simulate-data is an example on how to simulate the data presented in the paper.

diff --git a/README.md b/README.md
index 2bf32615263a347c4a555cc8b7859476bf8c962f..9e99961631efa026c5ac7a1f1e4c4c0c4391e2cd 100644
--- a/README.md
+++ b/README.md
@@ -4,41 +4,51 @@ This code implements the disentangled sticky HDP-HMM (DS-HDP-HMM) and two baseli

 ## Installation

 The code is written in Python 3.6. In addition to standard scientific Python libraries (numpy, scipy, matplotlib), the code expects the munkres package. You can install the munkres package using: `conda install -c conda-forge munkres`

 To download this code, run `git clone https://github.com/zhd96/ds-hdp-hmm.git`

 ## Examples

 There are examples for each model and emission function in the examples folder. We are working on more modulated code.

 You can run the example through `nohup python file-name.py command-line-parameters &` in general. See each example file for the specific command line parameters it uses. For examples, the files in examples/ds-hdp-hmm folder are examples for DS-HDP-HMM with the following different observations and samplers:

| File name                                       | Description                                                 |
| ----------------------------------------------- | ----------------------------------------------------------- |
| run_full_bayesian_gibbs_gaussian                | Gaussian observation, direct assignment sampler             |
| run_full_bayesian_gibbs_multinomial             | multinomial observation, direct assignment sampler          |
| run_full_bayesian_gibbs_poisson                 | Poisson observation, direct assignment sampler              |
| run_full_bayesian_approx_parallel_gibbs_poisson | Poisson observation, weak-limit sampler in parallel         |
| run_full_bayesian_approx_parallel_gibbs_ar      | auto-regressive observation, weak-limit sampler in parallel |

 The files in examples/s-hdp-hmm and examples/hdp-hmm are defined for S-HDP-HMM and HDP-HMM respectively as in the examples/ds-hdp-hmm.

 The file in examples/simulate-data is an example on how to simulate the data presented in the paper.

### Hybrid observation framework

The `code/hybrid/` package provides an extensible DS-HDP-HMM sampler that can model continuous (Gaussian with Normal-Inverse-Wishart prior) and categorical (Dirichlet-multinomial) features simultaneously. The new sampler exposes:

`HybridEmissionModel` – couples Gaussian and multiple categorical emission components.

`HybridObservations` – wraps aligned continuous and categorical observation streams.

`DSHDPHMMHybridSampler` – runs the direct-assignment Gibbs sampler while managing stickiness and hierarchical priors.

This framework can be used to experiment with mixed-type data (e.g. continuous sensor values paired with multi-class annotations) without modifying the legacy one-dimensional Gaussian or purely multinomial implementations.

 ## Data

 You can find sample datasets in the data folder. The multinomial and Gaussian data are synthetic datasets. The i01_maze15_2d_data_100ms_sample_trials.npz is a segment of Poisson data in paper<sup>1,2</sup>. The bee_seq_data.npz is the sequence 4-6 data in paper<sup>3</sup> with auto-regressive observations.

 ## Reference

 If you use this code please cite the paper:

 Zhou, D., Gao, Y., Paninski, L. Disentangled sticky hierarchical Dirichlet process hidden Markov model. ECML 2020. https://arxiv.org/abs/2004.03019

 ## Data sources

 1. Pastalkova, E., Wang, Y., Mizuseki, K., Buzsaki, G.: Simultaneous extracellular recordings from left and right hippocampal areas ca1 and right entorhinal cortex from a rat performing a left/right alternation task and other behaviors. CRCNS.org (2015). https://doi.org/10.6080/K0KS6PHF28. (https://buzsakilab.nyumc.org/datasets/PastalkovaE/i01/i01_maze15_MS.001/)
 2. Pastalkova, E., Itskov, V., Amarasingham, A., Buzsaki, G.: Internally generated cell assembly sequences in the rat hippocampus. Science 321(5894), 1322–1327 (2008)
 3. Oh, S.M., Rehg, J.M., Balch, T., Dellaert, F.: Learning and inferring motion patterns using parametric segmental switching linear dynamic systems. International Journal of Computer Vision 77(1-3), 103–124 (2008) (https://www.cc.gatech.edu/~borg/ijcv_psslds/)

## Data

You can find sample datasets in the data folder. The multinomial and Gaussian data are synthetic datasets. The i01_maze15_2d_data_100ms_sample_trials.npz is a segment of Poisson data in paper<sup>1,2</sup>. The bee_seq_data.npz is the sequence 4-6 data in paper<sup>3</sup> with auto-regressive observations.

## Reference

If you use this code please cite the paper:

Zhou, D., Gao, Y., Paninski, L. Disentangled sticky hierarchical Dirichlet process hidden Markov model. ECML 2020. https://arxiv.org/abs/2004.03019

## Data sources

1. Pastalkova, E., Wang, Y., Mizuseki, K., Buzsaki, G.: Simultaneous extracellular recordings from left and right hippocampal areas ca1 and right entorhinal cortex from a rat performing a left/right alternation task and other behaviors. CRCNS.org (2015). https://doi.org/10.6080/K0KS6PHF28. (https://buzsakilab.nyumc.org/datasets/PastalkovaE/i01/i01_maze15_MS.001/)
2. Pastalkova, E., Itskov, V., Amarasingham, A., Buzsaki, G.: Internally generated cell assembly sequences in the rat hippocampus. Science 321(5894), 1322–1327 (2008)
3. Oh, S.M., Rehg, J.M., Balch, T., Dellaert, F.: Learning and inferring motion patterns using parametric segmental switching linear dynamic systems. International Journal of Computer Vision 77(1-3), 103–124 (2008) (https://www.cc.gatech.edu/~borg/ijcv_psslds/)

