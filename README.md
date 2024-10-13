# AttTransCFusion

Code for paper "Attention-Guided Fusion of Transformers and CNNs for Enhanced Medical Image Segmentation".



## Environment

- Please prepare an environment with Python 3.8.19, PyTorch 1.10.0, and CUDA 11.1



## DataSets

| Dataset      | References                                                   |
| ------------ | ------------------------------------------------------------ |
| CVC-ClinicDB | [Bernal et al. 2015](https://doi.org/10.1016/j.compmedimag.2015.02.007)<br/>[CVC-ClinicDB (kaggle.com)](https://www.kaggle.com/datasets/balraj98/cvcclinicdb) |
| CVC-300      | [Vázquez et al. 2017](https://doi.org/10.1155/2017/4037190)  |
| CVC-ColonDB  | [Bernal et al. 2012](https://doi.org/10.1016/j.patcog.2012.03.002)<br/>[Vázquez et al. 2017](https://doi.org/10.1155/2017/4037190) |
| Kvasir-SEG   | [Jha et al. 2020](https://doi.org/10.1007/978-3-030-37734-2_37)<br/>[GitHub - DebeshJha/Kvasir-SEG: Kvasir-SEG: A Segmented Polyp Dataset](https://github.com/DebeshJha/Kvasir-SEG) |
| ETIS-Larib   | [Silva et al. 2014](https://doi.org/10.1007/s11548-013-0926-3)<br/> |
| ISIC 2017    | [ Gutman et al. 2016](https://doi.org/10.48550/arXiv.1605.01397)<br/>(https://challenge.isic-archive.com/data/) |



## Train/Test

- Train

  ~~~
  python train.py
  ~~~

- Test

  ~~~
  python test.py
  ~~~

  

## References

- [TransFuse](https://github.com/Rayicer/TransFuse)
- [PVT v2](https://github.com/whai362/PVT)
