import model
from dataset_io import Dataset
import train
import time
import config
import json

if __name__ == "__main__":
    config.run_id = 105
    config.finetune1_epochs = 5
    config.finetune2_epochs = 5
    datasets = {}
    for var in (config.pretrain_vars + ["640"]):
        dataset_io = Dataset(var)
        datasets[var] = dataset_io
    M = model.Net()
    D = model.D()
    M = model.prep_model(M)
    D = model.prep_model(D)

    training_logs = {}
    for cycle in range(5):
        start_time = time.time()
        print(f"Cycle {cycle}:")
        for ind, var in enumerate(config.pretrain_vars):
            print(f"Training on {var}...")
            T = train.Trainer(datasets[var], M, D)
            _, _, _, M, D = T.train(True)
        end_time = time.time()
        print(f"Training Time cost: {end_time - start_time}")
        print("Evaluating...")
        T = train.Trainer(datasets["640"], M, D)
        PSNR, _ = T.inference(write_to_file=False)
        print(f"Cycle {cycle} PSNR: {PSNR}")
        training_logs[cycle] = PSNR
        with open(f"{config.experiments_dir}{config.run_id:03d}/ensemble_training.json", "w") as f:
            json.dump(training_logs, f)
