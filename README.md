# GFedBridge: Gradient-Aware Federated Bridge Diffusion for Hybrid-Supervised Multimodal Medical Image Segmentation
This is the test code of our IEEE Journal of Biomedical and Health Informatics (JBHI), 2027 submitted paper "GFedBridge: Gradient-Aware Federated Bridge Diffusion for Hybrid-Supervised Multimodal Medical Image Segmentation".

 > Abstract: Hybrid-supervised federated training remains challenging in multimodal medical image segmentation due to differences in imaging acquisition modalities and annotation availability across clients. Medical images acquired through different modalities, such as endoscopy, ultrasonography, dermoscopy, and retinal fundus photography, exhibit distinct characteristics, while participating clients may provide varying levels of supervision, including sparse annotations such as points, scribbles, or bounding boxes, as well as dense segmentation masks. Existing methods mostly rely on strategies that depend on dense annotations at least at one client for supervision, refinement, or guidance. Such approaches can introduce annotation-shape bias and limit scalability when dense labels are unavailable. We propose a two-stage federated framework that separates client-specific segmentation learning from collaborative mask refinement. The first stage learns client-specific segmentation representations from different medical imaging modalities using the available supervision, without requiring dense segmentation masks. The next stage refines these predictions using a reliability-guided Brownian diffusion bridge conditioned on the image, available annotations, and spatial reliability. At the server, we introduce a novel gradient-aware aggregation strategy that assigns weights to client updates based on client gradient information and the aggregated gradient of the server. Extensive experiments on public medical image datasets evaluate the proposed method across different acquisition modalities and supervision settings, demonstrating its applicability to heterogeneous federated medical image segmentation.

>
## Dependencies
- Python 3.10 or 3.11
- PyTorch 2.10
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

## Create environment and install packages
- `conda create -n GFedBridge python=3.10 -y`
- `conda activate GFedBridge`
- `pip install -r requirements.txt`

## Testing
- Download the datasets folder from [GoogleDrive](https://drive.google.com/file/d/1DrRiLPOoXoMlnXKsxSx4c52QxHGKZ559/view?usp=sharing) and place them in the project directory.

- The folder structure within `datasets` should be organized as follows:
```
datasets/
├── C1/
│   ├── test/
│   │   ├── image/
│   │   └── mask/
├── C2/
├── C3/
├── C4/

```

- Test the segmentation performance

```
$ python test.py
```
