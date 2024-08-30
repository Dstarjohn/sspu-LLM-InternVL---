# InternVL部署微调实践

InternVL 是一种用于多模态任务的深度学习模型，旨在处理和理解多种类型的数据输入，如图像和文本。它结合了视觉和语言模型，能够执行复杂的跨模态任务，比如图文匹配、图像描述生成等。通过整合视觉特征和语言信息，InternVL 可以在多模态领域取得更好的表现。

![](.\lx-image\1.png)

**Dynamic High Resolution：InternVL独特的预处理模块：动态高分辨率，是为了让ViT模型能够尽可能获取到更细节的图像信息，提高视觉特征的表达能力。对于输入的图片，首先resize成448的倍数，然后按照预定义的尺寸比例从图片上crop对应的区域**

**Pixel Shuffle在超分任务中是一个常见的操作，PyTorch中有官方实现，即nn.PixelShuffle(upscale_factor) 该类的作用就是将一个tensor中的元素值进行重排列，假设tensor维度为[B, C, H, W], PixelShuffle操作不仅可以改变tensor的通道数，也会改变特征图的大小。**

## InternVL 部署微调实践

微调InterenVL使用xtuner，部署InternVL使用lmdeploy。

```python
# 创建目录
cd /root
mkdir -p model

# 复制模型到指定路径下
cp -r /root/share/new_models/OpenGVLab/InternVL2-2B /root/model/

# 创建虚拟环境
conda create --name xtuner python=3.10 -y
# 激活虚拟环境（注意：后续的所有操作都需要在这个虚拟环境中进行）
conda activate xtuner
# 安装一些必要的库
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
# 安装其他依赖
apt install libaio-dev
pip install transformers==4.39.3
pip install streamlit==1.36.0

# 创建一个目录，用来存放源代码
mkdir -p /root/InternLM/code
cd /root/InternLM/code
git clone -b v0.1.23  https://github.com/InternLM/XTuner

# 安装deepspeed
cd /root/InternLM/code/XTuner
pip install -e '.[deepspeed]'
# 安装LMDeploy
pip install lmdeploy==0.5.3

##查看xtuner命令
xtuner version
xtuner help

## 首先让我们安装一下需要的包
pip install datasets matplotlib Pillow timm
## 把数据集复制到指定路径
cp -r /root/share/new_models/datasets/CLoT-cn /root/InternLM/datasets/
# 把指定图片数据复制到指定路径（建议自己随便找一张，我复制了两张）
cp InternLM/datasets/ex_images/007aPnLRgy1hau0i7mahhj30ci0elaea.jpg /root/InternLM/

#创建存放demo代码的文件路径和py文件
mkdir /root/InternLM/code
cd /root/InternLM/code/
touch /root/InternLM/code/test_lmdeploy.py

# 将下面代码复制到test_lmdeploy.py中
from lmdeploy import pipeline
from lmdeploy.vl import load_image

pipe = pipeline('/root/mymodels/InternVL2-2B')

image = load_image('/root/InternLM/004atEXYgy1gtf7753qz3j60wv0wyqa402.jpg')
response = pipe(('请你根据这张图片，讲一个脑洞大开的梗', image))
print(response.text)
# 运行代码
python3 test_lmdeploy.py
```



上述命令的截图如下：

![](.\lx-image\2.png)

这是我选取一张梗图和对应的json数据集如下：

![](.\lx-image\3.png)

json数据集中这张图的数据：

![](.\lx-image\4.png)

这是运行了test_lmdeploy.py脚本推理后的结果，我们可以看到InternVL2-2b模型就不能很好的解读这个梗，推理的时候可能会出现Aborted (core dumped)这个运行错误，资源问题，建议清理清理内存和删除掉不用的虚拟环境，这里我推理太慢了，并且占用34168的显存大小，后来申请开发机资源太慢了，就用云服务器接着执行下脚本。

![](.\lx-image\5.png)

我们开始准备用刚才的数据集InternVL2微调策略：

```python

# 为了高效训练，请确保数据格式为：
{
    "id": "000000033471",
    "image": ["coco/train2017/000000033471.jpg"], # 如果是纯文本，则该字段为 None 或者不存在
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      }
    ]
  }
```



修改config配置文件的Settings部分，XTuner下 InternVL的config，文件在： `/root/InternLM/code/XTuner/xtuner/configs/internvl/v2/internvl_v2_internlm2_2b_qlora_finetune.py`，修改后的完整文件内容如下，直接cv吧，**注意里面data_root路径和教程不一样，教程有问题的**：

```python
# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import AutoTokenizer

from xtuner.dataset import InternVL_V1_5_Dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.hooks import DatasetInfoHook
from xtuner.engine.runner import TrainLoop
from xtuner.model import InternVL_V1_5
from xtuner.utils import PROMPT_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
path = '/root/model/InternVL2-2B'

# Data
data_root = '/root/InternLM/datasets/'
data_path = data_root + 'ex_cn.json'
image_folder = data_root
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 6656

# Scheduler & Optimizer
batch_size = 4  # per_device
accumulative_counts = 4
dataloader_num_workers = 4
max_epochs = 6
optim_type = AdamW
# official 1024 -> 4e-5
lr = 2e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 1000
save_total_limit = 1  # Maximum checkpoints to keep (-1 means unlimited)

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
model = dict(
    type=InternVL_V1_5,
    model_path=path,
    freeze_llm=True,
    freeze_visual_encoder=True,
    quantization_llm=True,  # or False
    quantization_vit=False,  # or True and uncomment visual_encoder_lora
    # comment the following lines if you don't want to use Lora in llm
    llm_lora=dict(
        type=LoraConfig,
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        target_modules=None,
        task_type='CAUSAL_LM'),
    # uncomment the following lines if you don't want to use Lora in visual encoder # noqa
    # visual_encoder_lora=dict(
    #     type=LoraConfig, r=64, lora_alpha=16, lora_dropout=0.05,
    #     target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'])
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
llava_dataset = dict(
    type=InternVL_V1_5_Dataset,
    model_path=path,
    data_paths=data_path,
    image_folders=image_folder,
    template=prompt_template,
    max_length=max_length)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=llava_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=default_collate_fn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True)

custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
```



采用的是qlora，因为50%A100不够用

```python
cd XTuner

NPROC_PER_NODE=1 xtuner train /root/InternLM/code/XTuner/xtuner/configs/internvl/v2/internvl_v2_internlm2_2b_qlora_finetune.py  --work-dir /root/InternLM/work_dir/internvl_ft_run_8_filter  --deepspeed deepspeed_zero1
```



这个训练时间比较长，这里用了50%的 Nvidia A100 40GB显存的显卡做训练，差不多6个多小时，训练结束后，我们就可以在指定路径`internvl_ft_run_8_filter`下看到我们的训练后的权重文件iter_3000.pth。

接下来就是合并权重文件和模型转换

```python
cd /root/InternLM/code/XTuner
# transfer weights
python xtuner/configs/internvl/v1_5/convert_to_official.py xtuner/configs/internvl/v2/internvl_v2_internlm2_2b_qlora_finetune.py /root/InternLM/work_dir/internvl_ft_run_8_filter/iter_3000.pth /root/InternLM/InternVL2-2B/
```



转换后的文件保存在我们上面命令指定的路径**`/root/InternLM/InternVL2-2B/`**中，小伙伴可以去看看大小，接下来我们还是直接执行`test_lmdeploy.py`这个文件，但是要把里面的Model加载换成我们刚才转换好的路径

```python
# test_lmdeploy.py内容
from lmdeploy import pipeline
from lmdeploy.vl import load_image

pipe = pipeline('/root/model/InternVL2-2B')

image = load_image('/root/InternLM/datasets/ex_images/004atEXYgy1gpb3tsdolwj60y219fwkp02.jpg')
response = pipe(('请你根据这张图片，讲一个脑洞大开的梗', image))
print(response.text)

# cd到指定路径运行脚本
cd /root/InternLM/code
python3 test_lmdeploy.py

```

完成的流程就是这样，这里之所以没有截图，是因为，最近开发机资源紧张，并且多模态InternVL2-2B大模型微调训练耗时太久，在这之前，有过微调，胡总恶化，Agent智能体的搭建的小伙伴们肯定知晓，所以我并没有截图，后面资源放开，我们补充的，如果大家有什么疑问还是可以直接在群里或者是任何方式直接问，欢迎和大家一起讨论技术难点痛点。



## 总结

我们通过对比发现微调前模型将图片内容识别说出来，而微调后的模型输出明细有变化。语气明显带有“梗”的味道，感兴趣的小伙伴可以按照文档来操作一遍，尝试微调多模态大模型。当然这个教程的数据集是利用别人整理好的数据集。如果微调更好玩的模型，数据集的整理和预处理很重要的流程，很大程度上决定了模型的输出效果呢，希望能和大家们一起进步！

