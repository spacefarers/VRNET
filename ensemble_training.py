import model
from dataset_io import Dataset
import train
import time
import config
import json
from fire import Fire


def ensemble_training(run_id=101, finetune1_epochs=5, finetune2_epochs=5, ensemble_epochs=15):
    config.tags.append("ensemble_training")
    config.run_id = run_id
    config.finetune1_epochs = finetune1_epochs
    config.finetune2_epochs = finetune2_epochs
    datasets = {}
    for var in (config.pretrain_vars + ["640"]):
        dataset_io = Dataset(var,train_all_data=True)
        datasets[var] = dataset_io
    M = model.Net()
    D = model.D()
    M = model.prep_model(M)
    D = model.prep_model(D)

    training_logs = {}
    for cycle in range(ensemble_epochs):
        config.ensemble_iter = cycle
        start_time = time.time()
        print(f"Cycle {cycle}:")
        for ind, var in enumerate(config.pretrain_vars):
            print(f"Training on {var}...")
            T = train.Trainer(datasets[var], M, D)
            _, _, _, M, D = T.train(disable_jump=True)
        end_time = time.time()
        print(f"Training Time cost: {end_time - start_time}")
        print("Evaluating...")
        T = train.Trainer(datasets["640"], M, D)
        PSNR, _ = T.inference(write_to_file=True, disable_jump=True)
        config.log({"PSNR": PSNR})
        T.save_plot()
        print(f"Cycle {cycle} PSNR: {PSNR}")
        training_logs[cycle] = PSNR
        with open(f"{config.experiments_dir}{config.run_id:03d}/ensemble_training.json", "w") as f:
            json.dump(training_logs, f, indent=4)


if __name__ == "__main__":
    Fire(ensemble_training)
