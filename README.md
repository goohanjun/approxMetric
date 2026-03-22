# Approximation of Word Mover Distance using Neural Networks

This project approximates **Word Mover’s Distance (WMD)** using a neural network model.

> 📄 Paper: *An Approximation of Word Mover Distance Using a Neural Network Model*

---

## 🧠 Motivation

Word Mover’s Distance (WMD) is a powerful metric for measuring semantic similarity between sentences.

However:

- Exact computation requires solving an optimal transport problem  
- Time complexity is **O(n³ log n)**  
- Too slow for large-scale applications (retrieval, clustering, search)

---

## 💡 Key Idea

Instead of directly approximating WMD with heuristics (e.g., averaging bounds),  
we **learn the relative position of WMD between its lower and upper bounds**.

Given:


Lower Bound ≤ WMD ≤ Upper Bound


The model predicts:


r ∈ [0, 1]


and reconstructs WMD as:


WMD ≈ LB + r · (UB - LB)


→ This transforms the problem from **absolute regression → structured prediction**

---

## 🏗 Model

- **Input**: two sentences (word embedding sequences)  
- **Encoder**: Bi-directional GRU  
- **Interaction**: Attention mechanism  
- **Output**: scalar ratio `r ∈ [0,1]` (sigmoid)  

### Loss


L = MSE(log(r_pred), log(r_true))


### Properties

- Linear time complexity w.r.t. sentence length  
- Efficient approximation using precomputed bounds  

---

## 📊 Results (MSE)

| Lower Bound | Harmonic | Geometric | Arithmetic | **Proposed** | **Improvement (%)** |
|------------|---------:|----------:|-----------:|-------------:|-------------------:|
| RWMD       | 0.0304   | 0.0224    | 0.0209     | **0.0124**   | **40.78%** |
| ACT-3      | 0.0095   | 0.0236    | 0.0126     | **0.0060**   | 36.91% |
| ICT        | 0.0090   | 0.0238    | 0.0125     | **0.0059**   | 34.30% |

---

## 🔑 Key Observations

- The proposed model **significantly outperforms traditional ensemble methods**
- Up to **40% reduction in MSE**
- Gains hold across different lower bounds (RWMD, ACT-3, ICT)
- Even weak lower bounds (RWMD) become competitive when combined with learning

---

## ⚡ Efficiency

| Method        | Runtime (ms) |
|--------------|-------------:|
| WMD (exact)  | 93.6 |
| **Proposed** | **2.4** |

→ Only **~2.5% of exact computation time**, while improving accuracy

---

## 🧠 Why It Works

### Traditional methods


WMD ≈ average(LB, UB)


→ ignores structure

### Proposed method


learn position between bounds


→ leverages:

- relative structure of bounds  
- distribution of real distances  
- learned nonlinear mapping  

---

## 📌 Summary

> **Learning the relative position between bounds is more effective than averaging them.**

This approach:

- improves accuracy  
- reduces computation cost  
- scales linearly  
- generalizes across different bounds  
