## Operation 1

# hf_fs cat

URI: `hf://models/Tongyi-MAI/Z-Image-Turbo/README.md`
Path: `README.md`
Bytes: 8000

---
license: apache-2.0
language:
- en
pipeline_tag: text-to-image
library_name: diffusers
---


<h1 align="center">⚡️- Image<br><sub><sup>An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer</sup></sub></h1>

<div align="center">

[![Official Site](https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage)](https://tongyi-mai.github.io/Z-Image-blog/)&#160;
[![GitHub](https://img.shields.io/badge/GitHub-Z--Image-181717?logo=github&logoColor=white)](https://github.com/Tongyi-MAI/Z-Image)&#160;
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint-Z--Image--Turbo-yellow)](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)&#160;
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Online_Demo-Z--Image--Turbo-blue)](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo)&#160;
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Mobile_Demo-Z--Image--Turbo-red)](https://huggingface.co/spaces/akhaliq/Z-Image-Turbo)&#160;
[![ModelScope Model](https://img.shields.io/badge/🤖%20Checkpoint-Z--Image--Turbo-624aff)](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo)&#160;
[![ModelScope Space](https://img.shields.io/badge/🤖%20Online_Demo-Z--Image--Turbo-17c7a7)](https://www.modelscope.cn/aigc/imageGeneration?tab=advanced&versionId=469191&modelType=Checkpoint&sdVersion=Z_IMAGE_TURBO&modelUrl=modelscope%3A%2F%2FTongyi-MAI%2FZ-Image-Turbo%3Frevision%3Dmaster)&#160;
[![Art Gallery PDF](https://img.shields.io/badge/%F0%9F%96%BC%20Art_Gallery-PDF-ff69b4)](assets/Z-Image-Gallery.pdf)&#160;
[![Web Art Gallery](https://img.shields.io/badge/%F0%9F%8C%90%20Web_Art_Gallery-online-00bfff)](https://modelscope.cn/studios/Tongyi-MAI/Z-Image-Gallery/summary)&#160;
<a href="https://arxiv.org/abs/2511.22699" target="_blank"><img src="https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv" height="21px"></a>


Welcome to the official repository for the Z-Image（造相）project!

</div>



## ✨ Z-Image

Z-Image is a powerful and highly efficient image generation model family with **6B** parameters. Currently there are four variants:

- 🚀 **Z-Image-Turbo** – A distilled version of Z-Image that matches or exceeds leading competitors with only **8 NFEs** (Number of Function Evaluations). It offers **⚡️sub-second inference latency⚡️** on enterprise-grade H800 GPUs and fits comfortably within **16G VRAM consumer devices**. It excels in photorealistic image generation, bilingual text rendering (English & Chinese), and robust instruction adherence.

- 🎨 **Z-Image** – The foundation model behind Z-Image-Turbo. Z-Image focuses on **high-quality generation**, **rich aesthetics**, **strong diversity**, and **controllability**, well-suited for creative generation, **fine-tuning**, and downstream development. It supports a wide range of artistic styles, effective negative prompting, and high diversity across identities, poses, compositions, and layouts.

- 🧱 **Z-Image-Omni-Base** – The versatile foundation model capable of both **generation and editing tasks**. By releasing this checkpoint, we aim to unlock the full potential for community-driven fine-tuning and custom development, providing the most "raw" and diverse starting point for the open-source community.

- ✍️ **Z-Image-Edit** – A variant fine-tuned on Z-Image specifically for image editing tasks. It supports creative image-to-image generation with impressive instruction-following capabilities, allowing for precise edits based on natural language prompts.

### 📥 Model Zoo

| Model | Pre-Training | SFT | RL | Step | CFG | Task | Visual Quality | Diversity | Fine-Tunability | Hugging Face | ModelScope |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Z-Image-Omni-Base** | ✅ | ❌ | ❌ | 50 | ✅ | Gen. / Editing | Medium | High | Easy | *To be released* | *To be released* |
| **Z-Image** | ✅ | ✅ | ❌ | 50 | ✅ | Gen. | High | Medium | Easy | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint%20-Z--Image-yellow)](https://huggingface.co/Tongyi-MAI/Z-Image) <br> [![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Z--Image-blue)](https://huggingface.co/spaces/Tongyi-MAI/Z-Image) | [![ModelScope Model](https://img.shields.io/badge/🤖%20%20Checkpoint-Z--Image-624aff)](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image) <br> [![ModelScope Space](https://img.shields.io/badge/%F0%9F%A4%96%20Demo-Z--Image-17c7a7)](https://www.modelscope.cn/aigc/imageGeneration?tab=advanced&versionId=569345&modelType=Checkpoint&sdVersion=Z_IMAGE&modelUrl=modelscope%3A%2F%2FTongyi-MAI%2FZ-Image%3Frevision%3Dmaster) |
| **Z-Image-Turbo** | ✅ | ✅ | ✅ | 8 | ❌ | Gen. | Very High | Low | N/A | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint%20-Z--Image--Turbo-yellow)](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) <br> [![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Z--Image--Turbo-blue)](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo) | [![ModelScope Model](https://img.shields.io/badge/🤖%20%20Checkpoint-Z--Image--Turbo-624aff)](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo) <br> [![ModelScope Space](https://img.shields.io/badge/%F0%9F%A4%96%20Demo-Z--Image--Turbo-17c7a7)](https://www.modelscope.cn/aigc/imageGeneration?tab=advanced&versionId=469191&modelType=Checkpoint&sdVersion=Z_IMAGE_TURBO&modelUrl=modelscope%3A%2F%2FTongyi-MAI%2FZ-Image-Turbo%3Frevision%3Dmaster) |
| **Z-Image-Edit** | ✅ | ✅ | ❌ | 50 | ✅ | Editing | High | Medium | Easy | *To be released* | *To be released* |                                                                                                                                                                                                                                                                                           | *To be released*                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

### 🖼️ Showcase

📸 **Photorealistic Quality**: **Z-Image-Turbo** delivers strong photorealistic image generation while maintaining excellent aesthetic quality.

![Showcase of Z-Image on Photo-realistic image Generation](assets/showcase_realistic.png)

📖 **Accurate Bilingual Text Rendering**: **Z-Image-Turbo** excels at accurately rendering complex Chinese and English text.

![Showcase of Z-Image on Bilingual Text Rendering](assets/showcase_rendering.png)

💡  **Prompt Enhancing & Reasoning**: Prompt Enhancer empowers the model with reasoning capabilities, enabling it to transcend surface-level descriptions and tap into underlying world knowledge.

![reasoning.jpg](assets/reasoning.png)

🧠 **Creative Image Editing**: **Z-Image-Edit** shows a strong understanding of bilingual editing instructions, enabling imaginative and flexible image transformations.

![Showcase of Z-Image-Edit on Image Editing](assets/showcase_editing.png)

### 🏗️ Model Architecture
We adopt a **Scalable Single-Stream DiT** (S3-DiT) architecture. In this setup, text, visual semantic tokens, and image VAE tokens are concatenated at the sequence level to serve as a unified input stream, maximizing parameter efficiency compared to dual-stream approaches.

![Architecture of Z-Image and Z-Image-Edit](assets/architecture.webp)

### 📈 Performance
According to the Elo-based Human Preference Evaluation (on [*Alibaba AI Arena*](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=T2I)), Z-Image-Turbo shows highly competitive performance agai

Content truncated. Resume with offset 8000.