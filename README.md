## Learning Rotation and Reflection Equivariant Representations for Electrical Impedance Tomography Reconstruction

The code in this toolbox implements the "Learning Rotation and Reflection Equivariant Representations for Electrical Impedance Tomography Reconstruction". More specifically, it is detailed as follow.

## Training
Dataset EIDORS generated data is at [training&valid](https://drive.google.com/file/d/1F6v90evQkIxNyfYI34ezZGsR0I_tPks0/view?usp=drive_link) [testset](https://drive.google.com/file/d/16NsMTAI_Xmae6-sGrHK2THU-LdsKEcYj/view?usp=drive_link)

Real-world data: Two real-world data can be downloaded from [UEF2017](https://fips.fi/open-datasets/eit-datasets/2d-electrical-impedance-tomography-dataset/) for the 2D electrical impedance tomography dataset collected by the Finnish Inverse Problems Society at the University of Eastern Finland in 2017 (UEF2017) and [KTC2023](https://erepo.uef.fi/items/afb2ad37-525b-4ce9-86a7-0fb8af457a57) for the Kuopio Tomography Challenge 2023.

Put the data at `./data/` and run `python main.py --mode train`

## Test

The pretraining weight `best.pt` is at [pre-training weight](https://drive.google.com/file/d/1jtAgakysIMKA7605xgHY47dyrxJbYSNM/view?usp=drive_link)

Download the pretraining weight and put it to `./rre/`

`run python main.py --mode test`

The prediction will be at `./rre/`

## Contact Information:
If you encounter any bugs while using this code, please do not hesitate to contact us.

Shuaikai Shi [shuaikai.shi@ku.ac.ae](shuaikai.shi@ku.ac.ae)
