# PhyX: Does Your Model Have the "Wits" for Physical Reasoning?
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

![Reasoning](https://img.shields.io/badge/Task-Reasoning-red) 
![Multi-Modal](https://img.shields.io/badge/Task-Multi--Modal-red) 
![PhyX](https://img.shields.io/badge/Dataset-PhyX-blue)  
![Claude3.7-Sonnet](https://img.shields.io/badge/Model-Claude3.7--Sonnet-green) 
![GPT-o4-mini](https://img.shields.io/badge/Model-GPT--o4--mini-green) 
![Intern-VL](https://img.shields.io/badge/Model-Intern--VL-green)
![Kimi-VL](https://img.shields.io/badge/Model-Kimi--VL-green)
![MiniCPM](https://img.shields.io/badge/Model-MiniCPM-green)
![DeepSeek-R1](https://img.shields.io/badge/Model-DeepSeek--R1-green)
![GPT-o3-mini](https://img.shields.io/badge/Model-GPT--o3--mini-green)

Code for the paper "[PhyX: Does Your Model Have the "Wits" for Physical Reasoning?](https://huggingface.co/datasets/Cloudriver/PhyX)".

For more details, please refer to the project page with **dataset exploration and visualization tools**: [https://phyx-bench.github.io/](https://phyx-bench.github.io/).

[[🌐 Project Page](https://phyx-bench.github.io/)] [[📖 Paper](https://arxiv.org/abs/2505.15929)] [[🤗 Huggingface Dataset](https://huggingface.co/datasets/Cloudriver/PhyX)]  [[🌐 Blog (中文) (TBD)](https://github.com/NastyMarcus/PhyX)]

<p align="center">
    <img src="assets/PhyX_Logo.png" width="20%"> <br>
</p>

## 📖 Outlines
- [PhyX: Does Your Model Have the "Wits" for Physical Reasoning?](#phyx-does-your-model-have-the-wits-for-physical-reasoning)
  - [📖 Outlines](#-outlines)
  - [🔔 News](#-news)
  - [📝 About PhyX](#-about-phyx)
  - [🔮 Usage](#-usage)
    - [Dataset Versions](#dataset-versions)
    - [Evaluation on PhyX](#evaluation-on-phyx)
  - [✅ Cite](#-cite)
  - [❤️ Contributors](#️-contributors)

## 🔔 News
- **[2025.05.27]** 🎉 PhyX is officially supported by [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for easy evalution.
- **[2025.05.23]** 🚀 The [arXiv paper](https://arxiv.org/abs/2505.15929) is online!
- **[2025.05.21]** 🚀 We release the testmini set of PhyX at [Huggingface](https://huggingface.co/datasets/Cloudriver/PhyX) and the [evaluation code](https://github.com/NastyMarcus/PhyX)!


## 📝 About PhyX

PhyX is the first large-scale benchmark specifically designed to assess models' ability in physical reasoning through realistic, visually grounded scenarios.

PhyX includes 3,000 meticulously collected multimodal questions, covering 6 reasoning types across 25 sub-domains and 6 core domains: thermodynamics, electromagnetism, mechanics, modern physics, optics, and wave acoustics.

![Sample](assets/data_stat.png)

PhyX specializes in university-level challenging questions presented through realistic, high-fidelity visual scenarios. Unlike general-purpose benchmarks, our tasks require models to integrate visual cues with implicit physical laws, going beyond simple knowledge recall and demanding nuanced, context-driven inference. This design enables a rigorous evaluation of true multimodal reasoning about the physical world, exposing key limitations in current models’ capabilities when handling professional-level scientific problems.

![Sample](assets/data_dis.png)

PhyX consists of 3,000 visually grounded physics questions, carefully curated across six distinct physics domains:
- Mechanics (550)
- Electromagnetism (550)
- Thermodynamics (500)
- Wave/Acoustics (500)
- Optics (500)
- Modern Physics (400)

Data examples:

![Sample](assets/data_sample.png)

## 🔮 Usage

### Dataset Versions
PhyX contains two subsets: testmini (1,000 questions) and test (3,000 questions). Each subset includes 12 versions tailored for different evaluation settings:
| File Name                      | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `PhyX_mini.tsv`               | Open-ended (OE) questions with full original text and image                 |
| `PhyX_mini_MC.tsv`            | Multiple-choice (MC) version with original description and image            |
| `PhyX_mini_SIMPLY.tsv`        | OE version with simplified description                                      |
| `PhyX_mini_MC_SIMPLY.tsv`     | MC version with simplified description                                      |
| `PhyX_mini_IMG.tsv`           | OE version with image only (no description)                                 |
| `PhyX_mini_MC_IMG.tsv`        | MC version with image only                                                  |
| `PhyX_mini_TL.tsv`            | OE version with image converted to text (`image_caption` only)              |
| `PhyX_mini_TL_MC.tsv`         | MC version with text-only format                                            |
| `PhyX_mini_TL_SIMPLY.tsv`     | OE text-only version with simplified description                            |
| `PhyX_mini_TL_MC_SIMPLY.tsv`  | MC text-only version with simplified description                            |
| `PhyX_mini_TL_IMG.tsv`        | OE version with image converted to text and no original description         |
| `PhyX_mini_TL_MC_IMG.tsv`     | MC version with image converted to text and no original description         |

- 🔍 mini stands for the 1,000-questions testmini set; the full version with 3,000 samples will be released soon.
- MC: multiple-choice
- no MC: open-ended (OE)
- SIMPLY: simplified descriptions
- TL: text-only (image converted to image_caption)
- IMG: description removed (image only)

### Evaluation on PhyX

The evaluation code is implemented based on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), and we thank the authors for their efforts.

Please follow the [official readme](README_vlmeval.md) to create a pip/conda environment.

We use DeepSeek-V3 as the LLM-based judger, please set the `SiliconFlow_API_KEY` or `Deepseek_API` to use it.
The former one would employ the DeepSeek-V3 provied by SiliconFlow and latter one for official severs.

Alternatively, you can perform rule-based judgment, which is free.

To evaluate a VLM on PhyX, please refer to the examples in `examples/MLLM/`, such as:

```
#*********judge based on rules*********
python -u run.py --data PhyX_mini \
    --model GPT4o_20241120 \
    --judge-args '{"valid_type": "STR"}'



#*********deepseek v3 from siliconflow as judger*********
## export SiliconFlow_API_KEY=

# valid_type: STR or LLM
python -u run.py --data PhyX_mini \
    --model GPT4o_20241120 \
    --judge deepseek-v3-si --judge-args '{"valid_type": "LLM"}'


#*********official deepseek v3 as judger*********

## export Deepseek_API=
## export OPENAI_API_BASE="https://api.deepseek.com"

python -u run.py --data PhyX_mini \
    --model GPT4o_20241120 \
    --judge deepseek-v3 --judge-args '{"valid_type": "LLM"}'

```

This example shows how to evaluate `GPT4o_20241120` using DeepSeek-V3 as the judge.

Details for these parameters:

- `--data`: The dataset configuration to evaluate, e.g., `PhyX_mini_MC` for multiple-choice or `PhyX_mini` for open-ended.
- `--model`: The model to be evaluated. Please refer to [this link](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb ) for supported models.
- `--valid_type`: Judgment method — `LLM` for LLM-based evaluation or `STR` for rule-based matching.
- `--judge`: judger,  `deepseek-v3-si` for deepseek-v3 provided by SiliconFlow (set SiliconFlow_API_KEY) while `deepseek-v3` for official (set Deepseek_API and OPENAI_API_BASE="https://api.deepseek.com").

If you want to evaluate in text only mode, please set `PHYX_TEXT_ONLY=true`.

After running the evaluation, results will be saved in the `outputs` folder.


## ✅ Cite
If you find **PhyX** useful for your your research and applications, please kindly cite using this BibTeX:

```bibtex
@misc{shen2025phyxdoesmodelwits,
      title={PhyX: Does Your Model Have the "Wits" for Physical Reasoning?}, 
      author={Hui Shen and Taiqiang Wu and Qi Han and Yunta Hsieh and Jizhou Wang and Yuyue Zhang and Yuxin Cheng and Zijian Hao and Yuansheng Ni and Xin Wang and Zhongwei Wan and Kai Zhang and Wendong Xu and Jing Xiong and Ping Luo and Wenhu Chen and Chaofan Tao and Zhuoqing Mao and Ngai Wong},
      year={2025},
      eprint={2505.15929},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.15929}, 
}
```

## ❤️ Contributors
> *Hui Shen<sup>1, 2</sup>, Taiqiang Wu<sup>1</sup>, Qi Han<sup>3</sup>, Yunta Hsieh<sup>2</sup>, Jizhou Wang<sup>4</sup>, Yuyue Zhang<sup>3</sup>, Yuxin Cheng<sup>1</sup>, Zijian Hao<sup>3</sup>, Yuansheng Ni<sup>5</sup>, Xin Wang<sup>6</sup>, Zhongwei Wan<sup>6</sup>, Kai Zhang<sup>6</sup>, Wendong Xu<sup>1</sup>, Jing Xiong<sup>1</sup>, Ping Luo<sup>1</sup>, Wenhu Chen<sup>5</sup>, Chaofan Tao<sup>1</sup>, Z. Morley Mao<sup>2</sup>, Ngai Wong<sup>1</sup>.*

> *<sup>1</sup>The University of Hong Kong, <sup>2</sup>University of Michigan, <sup>3</sup>Independent, <sup>4</sup>University of Toronto, <sup>5</sup>University of Waterloo, <sup>6</sup>The Ohio State University.*
