# Motion-Smoothing-for-VLA-DP

Aiming to reduce inference latency and smooth executed actions in VLA and Diffusion based planning.

![Demo](assets/motion-smooting.png)

## Setup

- Old chunk  
$$
\mathbf{a}^{old} = \{a^{old}_0, \dots, a^{old}_{T-1}\}
$$

- New chunk  
$$
\mathbf{a}^{new} = \{a^{new}_0, \dots, a^{new}_{T-1}\}
$$

- Delay: $d$ steps  
- Blend window: $L$ steps  

---

## 1. Time Alignment

At blending start, new chunk index is:

$$
a^{new}_{d+k}, \quad k = 0, \dots, L-1
$$

---

## 2. Linear Blending

$$
\tilde{a}_{T-L+k}
=
(1-\alpha_k)\,a^{old}_{T-L+k}
+
\alpha_k\,a^{new}_{d+k}
$$

$$
\alpha_k = \frac{k+1}{L}
$$

---

## 3. EMA Smoothing

$$
a_t = (1-\epsilon)a_{t-1} + \epsilon \tilde{a}_t
$$

---

## 4. Closed Form (constant target)

$$
a_t = a^* + (a_0 - a^*)(1-\epsilon)^t
$$

---

## Summary

- $\alpha$: smooth chunk transition  
- $\epsilon$: temporal smoothing  