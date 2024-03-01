# Single-StyleBoost

## Official PyTorch implementation of the paper "Single-StyleBoost"

Recent advancements in text-to-image models, such as Stable Diffusion, have demonstrated their ability to synthesize visual images through natural language prompts. One approach of personalizing text-to-image models, exemplified by Dream-Booth, fine-tunes the pre-trained model by binding unique text identifiers with a few images of a specific subject. Although existing fine-tuning methods have demonstrated competence in rendering images according to the styles of famous painters, it is still challenging to learn to produce images encapsulating distinct art styles due to abstract and broad visual perceptions of stylistic attributes such as lines, shapes, textures, and colors. In this paper, we present a new fine-tuning method, called StyleBoost, that equips pre-trained text-to-image models to produce diverse images in specified styles from text prompts. By leveraging around 15 to 20 images of StyleRef and Aux images each, our approach establishes a foundational binding of a unique token identifier with a broad realm of the target style, where the Aux images are carefully selected to strengthen the binding. This dual-binding strategy grasps the essential concept of art styles and accelerates learning of diverse and comprehensive attributes of the target style. Experimental evaluation conducted on three distinct styles - realism art, SureB art, and anime - demonstrates substantial improvements in both the quality of generated images and the perceptual fidelity metrics, such as FID and CLIP scores.

![over_fig3](https://github.com/matrix215/Single-StyleBoost/assets/101815603/e5676700-8ec2-46e4-82b5-4924da8b0852)


## Repository Overview


## Citation
```bash
@INPROCEEDINGS{10392676,
  author={Park, Junseo and Ko, Beomseok and Jang, Hyeryung},
  booktitle={2023 14th International Conference on Information and Communication Technology Convergence (ICTC)}, 
  title={StyleBoost: A Study of Personalizing Text-to-Image Generation in Any Style using DreamBooth}, 
  year={2023},
  volume={},
  number={},
  pages={93-98},
  keywords={Training;Measurement;Visualization;Art;Shape;Natural languages;Transforms;text-to-image models;diffusion models;person-alization;fine-tuning},
  doi={10.1109/ICTC58733.2023.10392676}}
```
