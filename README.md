# RK4_Pi_Block
Learning Spatiotemporal Dynamics from Sparse Data via a High-order Physics-encoded Network

The paper link is coming soon.

---

## Overview

We present a novel high-order physics-encoded learning framework for capturing the intricate dynamical patterns of spatiotemporal systems from limited sensor measurements. Our approach centers on a deep convolutional-recurrent network, which hard encodes known physical laws (e.g., PDE structure and boundary conditions) into the learning architecture. Moreover, the high-order time marching scheme (e.g., Runge-Kutta fourth-order) is introduced to model the temporal evolution. We conduct comprehensive numerical experiments on a variety of complex systems to evaluate our proposed approach against baseline algorithms across two tasks: reconstructing high-fidelity data and identifying unknown system coefficients. We also assess the performance of our method under various noisy levels and using different finite difference kernels. The comparative results demonstrate the superiority, robustness, and stability of our framework in addressing these critical challenges in SciML.

### Highlights

- This paper focuses on PDE inverse analysis in the context of coefficient identification and high-resolution dynamics reconstruction.

- The known physical laws are encoded into the network using a -block, and a high-order time marching scheme is considered for modeling temporal evolution.

- Comprehensive experiments have been performed to validate the superiority, robustness, and stability of our approach against baseline models across various PDE systems. 

---

## System Requirements and Usage

### Hardware requirements
 
All the experiments are performed on an NVIDIA A100 GPU Card.

### Environment requirements

Install the required dependencies:

```shell
pip install -r requirements.txt
```

### Dataset

The dataset is provided via a [Google Drive link](https://drive.google.com/drive/folders/1T14u26rcJc5xkir4LTO_auo7G9Zjke_K?usp=sharing). 

### Implementations

1. Train the RK4-Pi-Block with [Neptune](https://neptune.ai/):

    ```shell
    sh train.sh
    ```

2. Evaluate the best model:

    ```shell
    sh eval.sh
    ```

## License

This project is released under the GNU General Public License v3.0.
