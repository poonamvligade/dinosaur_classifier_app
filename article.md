## BackStory

My son is fascinated with all things about dinosaurs.
So, I built a 🦖 🦕 classifier for him as homework for the first week of the fastai 2022 cohort.
Choose from examples or upload image of an dinosaur from below categories to find out its name.

## Collecting Dinosaur Images

200 images from each of the below categories were downloaded from duckduckgo inspired by Jeremy Howard Kaggle [kernel](https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data).

- Pteradactyl 
- Parasaurolophus
- Triceratops
- Brachiosaurus
- Stegosaurus
- T-rex
- Gigantosaurus

### Datablock code 

```
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(256)],
    batch_tfms=aug_transforms(size=224),
).dataloaders(path)

dls.show_batch(max_n=12)
```

## Training 

A `seresnext50_32x4d` model was finetuned on the dataset for 20 epochs. 
Since the dataset was complex, adding data augmentation to the datablock improved the results. 



### Finetuning 


|epoch|train_loss|valid_loss|error_rate|time|
|-------|---------------|---------------|---------------|-------|
|0|2.356192|1.444657|0.443439|00:10


|epoch|train_loss|valid_loss|error_rate|time|
|-------|---------------|---------------|---------------|-------|
0|1.080438|1.082863|0.343891|00:08
1|1.016290|1.013307|0.316742|00:07
2|0.920574|0.907412|0.271493|00:07
3|0.849076|0.957548|0.307692|00:07
4|0.76567|0.968153|0.285068|00:07
5|0.673238|0.935877|0.257919|00:07
6|0.604583|1.019347|0.244344|00:07
7|0.558268|0.998316|0.248869|00:07
8|0.510839|1.005702|0.262443|00:07
9|0.477934|1.021464|0.266968|00:07
10|0.418508|1.072529|0.285068|00:07
11|0.388544|0.980327|0.266968|00:07
12|0.358420|0.981039|0.266968|00:07
13|0.335584|0.984379|0.262443|00:07
14|0.327304|0.995534|0.262443|00:07
15|0.299455|0.973741|0.271493|00:07
16|0.279585|0.968682|0.248869|00:08
17|0.250505|0.958490|0.253394|00:07
18|0.241029|0.962064|0.257919|00:07
19|0.228444|0.962610|0.262443|00:07

## Observations

- The dataset is very noisy, and is different from imagenet
- The model finds hard to differntiate between `gigantosaurus` and `T-rex` since they are so similar.


## About JarvisLabs

This app has been hosted on JarvisLabs.ai

Jarvislabs provides a 1-click GPU cloud platform for your Deep Learning training. 
Spin up modern Nvidia GPUs like A100, A6000, A5000, Quadro RTX 5000, Quadro RTX 6000, and more within seconds.
Get instant access to JupyterLab or plugin VS code through SSH.

## Note by Author 

Hope you liked the app. Please feel free to contact me if you have any suggestions on improving this app.

[Poonam Ligade](https://twitter.com/Poonamligade)

[How to deploy an AI App on JarvisLabs.ai](https://www.youtube.com/watch?v=_A9aIzsHN3Y)

Enjoy building your own app!




