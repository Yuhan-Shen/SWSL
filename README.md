# SWSL

We provide the implementation of [Semi-Weakly-Supervised Learning of Complex Actions from Instructional Task Videos](https://openaccess.thecvf.com/content/CVPR2022/papers/Shen_Semi-Weakly-Supervised_Learning_of_Complex_Actions_From_Instructional_Task_Videos_CVPR_2022_paper.pdf) based on [MuCon](https://github.com/yassersouri/MuCon).

## Preparation
Please follow the instructions in  [MuCon](https://github.com/yassersouri/MuCon) for environment installation and data preparation.

Download the files of video lists for different ratios of weakly-labeled videos and put them in the dataset folder. TODO: update download link.


## Running
Please run this command to train and test the model:
```
 python src/train_test_swsl.py --cfg src/configs/docker/inside.yaml --set add_dataset.ratio 0.1  --set dataset.split 1
```
You may change the ratio of weakly-labeled videos and dataset split. You may also change other parameters defined in ```src/configs/mucon/default.py```.

## Citation
If you find the project helpful, we would appreciate if you cite the work:

```
@article{Shen-SWSL:CVPR22,  
         author = {Y.~Shen and E.~Elhamifar},  
         title = {Semi-Weakly-Supervised Learning of Complex Actions from Instructional Task Videos},  
         journal = {{IEEE} Conference on Computer Vision and Pattern Recognition},  
         year = {2022}}
```


## Contact
shen [dot] yuh [at] northeastern [dot] edu


## Acknowledgement
The code-base is built using the [fandak](https://github.com/yassersouri/fandak) library and [MuCon](https://github.com/yassersouri/MuCon) repo.

