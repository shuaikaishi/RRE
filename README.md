## Learning Rotation and Reflection Equivariant Representations for Electrical Impedance Tomography Reconstruction

The code in this toolbox implements the "Learning Rotation and Reflection Equivariant Representations for Electrical Impedance Tomography Reconstruction". More specifically, it is detailed as follow.

## Training

Put the data at ./data/ and run `python main.py --mode train`

## Test

The pretraining weight best.pt is at [pre-training weight](https://drive.google.com/file/d/1jtAgakysIMKA7605xgHY47dyrxJbYSNM/view?usp=drive_link)

Download the pretraining weight and put it to `./rre/`

`run python main.py --mode test`

The prediction will be at `./rre/`

## Contact Information:
If you encounter any bugs while using this code, please do not hesitate to contact us.

Shuaikai Shi [shuaikai.shi@ku.ac.ae](shuaikai.shi@ku.ac.ae)
