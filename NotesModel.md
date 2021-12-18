

# Push-pull

## A Push-Pull Gradient Method for Distributed Optimization in Networks

***2020 59th IEEE Conference on Decision and Control (CDC)***

***DOI: 10.1109/CDC.2018.8619047***

- Shi Pu

---

The algorithm uses a row stochastic matrix for the mixing of the decision variables, while it employs a column stochastic matrix for tracking the average gradients.

## Push–Pull Gradient Methods for Distributed Optimization in Networks

## A_Robust_Gradient_Tracking_Method_for_Distributed_Optimization_over_Directed_Networks

***2020 59th IEEE Conference on Decision and Control (CDC)***

- Shi Pu

---

### 1.Motivations

**Background**: The domains of transport and logistics are increasingly relying on autonomous mobile robots to handle and distribute passengers or resources.

**Challenges**:

- **Balance** between <font color="blue">optimality</font> and real-world <font color="blue">efficiency</font> i. e. <font color="red">centralized</font> vs <font color="red">ecentralized</font>: 

- **Centralized** **methods** require central units to gather information from all robots and organize the optimal path for each of them, consuming large computational resources.

- As the system scales, **decentralized approaches** become increasingly popular, where each robot estimates or communicates others' future trajectories via broadcasting or distance-based communication. But the <font color="blue">optimality</font> of the solutions and the overall <font color="blue">resilience</font> of the team to disruptions are affected due to the following limitations:  

  - <font color="red">Redundant information</font> when communication happens concurrently and equivalently among many neighboring robots
  
  - <font color="blue">Robust</font> and <font color="blue">continuous</font> communication cannot yet be guaranteed due to <font color=red>limited bandwidth</font>, <font color=red>large data volumes</font>, and <font color=red>interference</font> from the surroundings
  - The <font color="blue">convergence</font> of the coordination process is hard to be ensured under a fully decentralized frame work <font color="red">without any priority of planning</font>
  
- Permutation equivariance, time invariance in <font color="red">dynamic</font> communication

- <font color="blue">Generalization</font> of path planning methods

**New trends**:

*Communication-aware path planning approaches* by explicitly considering communication efficiency during path generation and path optimization addressing to whom and when

### 2.Related work

<table>
    <caption><b>Comparison of related methods</b></caption>
	<tr>
	    <th align="center">Model</th>
	    <th align="center">Methods</th>
	    <th align="center">Limitations</th>  
	</tr >
	<tr >
	    <th align="center" rowspan="2">DNN</td>
    	<td><font size="2">A decentralized multi-agent collision avoidance algorithm based on <b>Deep Reinforcement Learning</b></font><br/><font size="1"><i>Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcemrnt Learning</i></font></th>
		<td rowspan="2"><font size="2">1.<font color="red">No inter-robot communication</font>.<br>2.Not exploit the full potentials of the decentralized system</font></td>
	</tr>
	<tr >
	    <td><font size="2"><b>PRIMAL</b>: multi-agent path-finding that integrated <b>Imitation Learning</b> (via an expert algorithm) and multi-agent <b>RL</b><br/><font size="1"><i>PRIMAL: Pathfinding via Reinforcement and Imitation Multi-Agent Learning</i></font></font></th>
	</tr>
	<tr >
	    <th align="center">CNN+<br>GNN</td>
	    <td><font size="2">CNN extracts adequate features from local observations, GNN learns to communicate these features among robots</font><br><font size="1"><i>Graph Neural Networks for Decentralized Multi-Robot Path Planning</i></font></td>
	    <td><font size="2">Large-scale graphs with noisy and <font color="red">redundant information</font></td>
	</tr>
	<tr >
	    <th align="center">GAT~</td>
	    <td><font size="2">A learning-based communication model that constructed the communication group on a <b>static graph</b><br><font size="1"><i>When2com:Multi-Agent Perception via COmmunication Graph Grouping</i></font></td>
	    <td><font size="2">Large-scale graphs with noisy and <font color="red">redundant information</font></td>
	</tr>
	<tr >
	    <th align="center">MAGAT</td>
	    <td><font size="2">Message-Aware Graph Attention Networks<br><font size="1"><i>Message-Aware Graph Attention Networks for Large-Scale Multi-Robot Path Planning</i></font></td>
	    <td><font size="2">---</font></td>
	</tr>
</table>
### 3.Problem Formulation

#### Problem

<div align=center>
<img src="problem formulation.png" alt="solutions" style="zoom:80%; float:right;" />
</div>

- 2D grid world $\mathcal W$
- static obstacles $\mathcal C\subset\mathcal W$
- robots set $\mathcal V=\{\mathcal v_1,…,\mathcal v_N\}$
- action $\tilde {\pmb u}_t^i$
- a sequential decision-making problem

#### Assumptions
- No global positioning, field of view (**FOV**)
- The goal outside the FOV is clipped to the FOV boundary in a local reference frame
- Communication happens **instantly** compared with the robot’s movement, and unblocked
- **Limit the communication**: robots can only send out their features at a specific bandwidth, but cannot access the ownership of the feature received
#### Communications

- Dynamic distance-based communication network $\mathcal G_t=(\mathcal V,\mathcal E_t,\mathcal W_t)$
  - $\mathcal V$ is the set of robots
  - $\mathcal E_t\subseteq\mathcal V\times\mathcal V$ is the set of edges
  - $\mathcal W_t:\mathcal E_t\to\mathbb R$ is a function assigning weights to the edges

- Two robots $v_i$ and $v_j$ can communicate with each other if $\Vert \pmb {\rm p}_i-\pmb {\rm p}_j\Vert \le r_{\rm COMM}$
- An adjacency matrix ${\rm S}_t\in\mathbb R^{N\times N}$, where $[{\rm S}_t]_{ij}=s_t^{ij}=\{0,1\}$

- The corresponding edge weight $\mathcal W_t(v_i,v_j)=w_t^{ij}=[{\rm S}]_{ij}[{E}]_{ij}\in[0,1]$
  - $[{\rm S}]_{ij}$ represents the graph connectivity
  - $[{E}]_{ij}$ indicates relative importance (attention) of the information contained in the messages

### 4.Preliminaries

- <font color="blue">**Observation Matrix**</font>

  $N$ agents, and each robots have $F$ observtions

$$
\mathbf{X}_t = 
\begin{bmatrix} ({\rm \tilde x}_t^1)^\intercal \\ \vdots \\ ({\rm \tilde x}_t^N)^\intercal \end{bmatrix} =
\begin{bmatrix} {\rm  x}_t^1 & \cdots & {\rm x}_t^F \end{bmatrix}
\in \mathbb R^{N\times F}
\tag{1} \label{eq1}
$$

- <font color="blue">**Graph Shift Operation (GSO)**</font>

  The convolution on graph features is defined by ==linear combination== of neighboring node feature values ${\rm X}_t $ from the graph $\mathcal G_t$.

  The value at node $i$ forf eature $f$ is
$$
[\mathbf{S}_t\mathbf{X}_t]_{if} = 
\sum_{j=1}^N [\mathbf{S}_t]_{if}[\mathbf{X}_t]_{if} =
\sum_{j:v_j\in \mathcal N_i}s_t^{ij}x_t^{jf}
\tag{2} \label{eq2}
$$

- <font color="blue">**Graph Convolution**</font>

  A linear combination of shifted versions of the signal from the graph:
  $$
  \mathcal A(\mathbf{X}_t;\mathbf{S}_t) = 
  \sum_{k=0}^{K-1}\mathbf{S}_t^k \mathbf{X}_t \mathbf{A}_k
  \tag{3} \label{eq3}
  $$
  
  - $\{\mathbf{A}_k\}\in \mathbb R^{F\times G}$  represents the ==filter coefficients== combining different observations
  - $k$ hop neighbors
  -  $F$ : input dimension
  -  $G$ : output dimension


  - <font color="blue">**Graph Neural Networks**</font>

    An GNN module is a cascade $L$ layers graph convolutions in $\eqref{eq3} $
    $$
    \mathbf{X}_{\ell}=\sigma\left[\mathcal{A}_{\ell}\left(\mathbf{X}_{\ell-1} ; \mathbf{S}\right)\right] \text {\quad for \quad} \ell=1, \ldots, L
    \tag{4} \label{eq4}
    $$

    - $\sigma : \mathbb R \to \mathbb R$ : point-wise non-linearity (==activation function==)
    - $\mathbf{S}=\mathbf{S}_t$ corresponding to the communication network at time $t$
    - The first layer $F_0=F$
    - The last layer $F_L=G$ is the fused information via ==multi-hop communication== and ==multi-layer convolutions==, which will be used to predict action $\mathbf{U}_t$ at time $t$

- <font color="blue">**Graph Attention neTwork**</font>
  $$
  \mathcal A(\mathbf{X}_t;\mathbf{S}_t) = 
  \sum_{k=0}^{K-1}(\mathbf{E} \odot \mathbf{S}_t)^k \mathbf{X}_t \mathbf{A}_k
  \tag{5} \label{eq5}
  $$

  - $\mathbf{E}$ is an attention matrix of the same dimensions as $\mathbf S$, which is written as a normalization form (softmax function):

  $$
  [E]_{ij} = \frac {\text {exp}(LeakyReLU(e_{ij}))}{\sum_{k\in \mathcal N_i} \text {exp}(LeakyReLU(e_{ik}))}
  \in [0,1]
  \tag{6} \label{eq6}
  $$
  
  - $\mathcal N_i$ is the collection of all the neighboring nodes
  
  $$
  e_{ij}=((\tilde{\mathrm{x}}_t^i)^\intercal\mathbf{A}_k\Vert(\tilde{\mathrm{x}}_t^j)^\intercal\mathbf{A}_k)\mathbf{H}
  \tag{7} \label{eq7}
  $$
  
  - $\mathbf{H}$ is a $2G_l\times 1$ matrix
  - $\Vert$ represents concatenation
  
  $$
  \mathbf{X}_\ell=\Vert_{p=1}^P(\sigma[\mathcal{A}_{\ell}^p(\mathbf{X}_{\ell-1} ; \mathbf{S})]) \text {\quad for \quad} \ell=1, \ldots, L
  \tag{8} \label{eq8}
  $$
  
  - $P$ is the number of independent heads in the layer
  
  <font color="red">**Notation**</font>: the trainable weight matrix $\mathbf{A}_k$ serves both in the attention weight computation $\eqref{eq7}$ and the feature aggregation $\eqref{eq5}$ 
  
- <font color="blue">**Messag-Aware Graph Attention neTwork**</font>
  
  Similar to GNN and GAT
  $$
  e_{ij}=(\tilde{\mathrm{x}}_t^i )^\intercal W\tilde{\mathrm{x}}_t^j
  \tag{9} \label{eq9}
  $$
  
  - $W$ is a weight matrix serving as a ==key-query-like attention==
  - each trainable weight matrix $W$ in each attention head $p$ and each layer $l$ is independent
  
- <font color="blue">**key-query-like attention mechanism**</font>

  [Multi-headed self-attention](https://zhuanlan.zhihu.com/p/365386753)

  Or refer to <font size="2" color="purple">Multi-headed Self-attention（多头自注意力）机制介绍 - 知乎.pdf</font>

### 5.Solution

<img src="solution scheme.png" alt="solutions" style="zoom:100%; float:center;" />

- A. Dataset creation
- B. Processing observations
- C. Network architecture
  - CNN-based perception
  - Graph-based communication
- Action policy
- Training and online expert

  $$
  \hat{\theta}=\underset{\theta}  {\operatorname{argmin}}\sum_{\left(\mathbf{Z}_{t}^{i}, \mathbf{U}_{t}^{* i}\right) \in \mathcal{T}}\mathcal{L}\left(\mathbf{U}_{t}^{*},\mathcal{F}\left(\mathbf{Z}_{t}^{i},\mathcal{G}_{t}\left(\mathbf{Z}_{t}^{i}\right)\right)\right)\tag{17} \label{eq17}
  $$
  - $\mathbf{Z}_{t}^{i}$ is the processed observation
  - $\mathbf{U}_{t}^{* i}$ is the expert action at this situation
  - $\mathcal L$ is the loss function
  - the training objective is to obtain a classifier $\mathcal{F}$ with trainable parameters $\theta$ given the training dataset $\mathcal{T}$

### 6.Contributions

- The paper combines a **Graph Neural Network (GNN)** with a **key-query-like attention mechanism** to improve the effectiveness of inter-robot communication and demonstrates the suitability of applying the model on **dynamic communication graphs** by proving its *permutation equivariance* and *time invariance* property.
- The paper investigates the impact of **reduced communication bandwidth** by reducing the size of the shared features, and deploy a **skip-connection bottleneck structure** to preserve self-information and maintain model performance.
- The paper demonstrates the **generalizability** of the model by training the model on small problem instances and testing it on increasing robot density, varying map size, and much larger problem instances (more than 100 robots). ==The **MAGAT** model is shown to be more efficient in learning general knowledge of path planning as it achieves better **generalization performance** than the baseline systems under various scenarios==.

### 7.Limitations

- Supervised learning methods to optimize goals 

- Discrete action (up, down, left, right, and idle), grid game



















