The post-training **FLOAT32** and **INT8** quantization applied on the Sequential and Siamese models. The following are the results:


**SIZE**:

| Model Name | Siamese | Sequential |
| ------ | ------ | ------ |
| Original | 258,998 KB| 2,360 KB|
| FLOAT32 |  64,751 KB| 201 KB |
| INT8    |  64,797 KB|  214 KB |

**ACCURACY**:

| Model Name | Siamese | Sequential |
| ------ | ------ | ------ |
| Original | 89 - 90 % | 59 %  |
| FLOAT32 |  59.8756 %| 58.6314 % |
| INT8    |  18.9736 %|  48.056 %|

**TIME TO PREDICT**:
The performance time includes the loading models and loading test dataset time. 

1. Sequential FLOAT32 TFlite model:

![A1](/uploads/3cdbed0a2db006390dec632a7e865202/A1.PNG)

2. Sequential INT8 TFlite model:

![A2](/uploads/f865c01288d26fc22be6c8d01d81fa8f/A2.PNG)

3. Siamese FLOAT32 TFlite model:

![B1](/uploads/4beec86241f262c8159af661f459eee6/B1.PNG)

4. Siamese INT8 TFlite model:

![B2](/uploads/d4dfee2eb1640bb92ebfedc93a2b83ff/B2.PNG)