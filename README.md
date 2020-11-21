# -
DataFountain 通用音频分类

https://www.datafountain.cn/competitions/486

将wav文件转为频谱，用基于Conv1D的残差网络做分类。

下载数据，将训练集和测试集分别解压到 data/train/ 和 data/test/ 下。

运行程序

```
cd ./code

python ResidualNetwork.py
```

测试结果保存在result.csv
