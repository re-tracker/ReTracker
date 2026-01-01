
# API

- 给一组有序的images, 通过tracking得到[tracks]；
- tracks转化成合理的模型；

1 优化了训练流程和显存消耗，现在训练时支持更多点数;

2 支持在不同分辨率下推理;

3 添加了在线tracking demo;

4 支持多GPU调用;

5 更新backbone: 使用pretrained backbone并不再训练backbone(获得更快的训练速度)
(我们没有观察到v3backbone带来效果上的提升;可能由于只用到了高层信息, 底层信息主要由cnn提供;)
6 支持RXXX在线视频流输入的online demo;

7 更新了在线推理GUI;

8 更新了快速推理version; 

9 更多对应点:支持输出查询点周围semi-dense的预测;


TODO 使用64帧数据集训练(现在24帧);
TODO 添加更多训练数据集, 
TODO 提高tracking训练数据集分辨率(现在为256*256);这限制了模型在大分辨率推理时的效果;
TODO 压缩加速模型;
TODO 增加depth以实现3D空间上的track;
# Roadmap

Moved from the repo root for readability.
