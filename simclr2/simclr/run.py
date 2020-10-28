from simclr import SimCLR
import yaml
import os
from dataset_wrapper import DataSetWrapper

model_path = os.environ["RESULT_DIR"]
print ('model_path: ', model_path)


def main():
    config = yaml.load(open(model_path+"/_submitted_code/simclr/config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    simclr = SimCLR(dataset, config)
    simclr.train()


if __name__ == "__main__":
    main()
